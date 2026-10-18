# In-Instance Question Adaptation (Few-Shot Head Calibration)

Bio-Cascade SLM v15.1 provides native support for **in-instance question adaptation** via the method `BioCascadeGenerativeDecoder.adapt_to_question()`.

This feature allows any active instance of `BioCascadeGenerativeDecoder` running in memory to be specialized in-place for a specific user-defined question or business classification task in **~2 seconds on a commodity CPU**, using as few as 5–20 labeled examples.

---

## 💡 Why In-Instance Adaptation?

Standard foundation models (LLMs) typically require:
1. Costly parameter fine-tuning (LoRA, full fine-tuning) requiring GPU clusters and hours of training.
2. In-context learning (few-shot prompting) which inflates prompt token length, increases latency proportionally, and remains susceptible to stochastic drift.

**Bio-Cascade SLM** takes an architectural approach grounded in its Cascade-Correlation hierarchy:
- Layers **L0 through L5** (BPE token embeddings, Fourier Spectral Wave Memory, and deep cascade dense representations) are **frozen**. They act as an invariant, generalized feature extractor.
- Layer **L6** (the 32-dimensional verified salience routing head) is adapted in-place using PyTorch AdamW optimization with calibrated cross-entropy loss and class-frequency weighting.
- Training completes in **1–3 seconds on a single CPU core** (no GPU required).
- The adapted instance retains 100% of its native speed (15–25 ms per query) with zero token-overhead during subsequent inference.

---

## 🛠️ Method Signature & Parameters

```python
def adapt_to_question(
    self,
    question: str,
    data: Any,
    epochs: int = 150,
    lr: float = 0.005,
    weight_decay: float = 1e-5,
    yes_weight: float = 1.5,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `question` | `str` | *(required)* | The specific target question against which texts will be verified (e.g. *"Did someone use AI or deepfakes to steal money or commit fraud in this story?"*). |
| `data` | `Any` | *(required)* | The training dataset containing text samples and target decisions. Supports multiple data formats (see below). |
| `epochs` | `int` | `150` | Number of gradient descent optimization steps on the 32-D routing head. 100–150 epochs typically converge in 1.5–2.5 seconds on CPU. |
| `lr` | `float` | `0.005` | Learning rate for the AdamW optimizer. |
| `weight_decay` | `float` | `1e-5` | L2 weight decay regularizer preventing overfitting on small sample counts. |
| `yes_weight` | `float` | `1.5` | Positive class loss multiplier in the cross-entropy formulation. Values between `1.2` and `1.8` ensure high recall while preventing false positive drift on hard negatives. |
| `save_path` | `str \| Path \| None` | `None` | Optional path where the adapted model checkpoint (`.pkl`) will be saved upon completion. If `None`, the model remains updated in-place in active memory. |

### Return Value

Returns a dictionary containing adaptation telemetry and convergence statistics:
```python
{
    "status": "success",
    "question": str,
    "samples_count": int,
    "positive_samples": int,
    "negative_samples": int,
    "epochs": int,
    "train_accuracy_pct": float,
    "false_negatives": int,
    "saved_checkpoint": str | None
}
```

---

## 📊 Supported Input Data Formats

`adapt_to_question()` automatically inspects and parses any of the following standard data formats:

### 1. List of Dictionaries
```python
data = [
    {
        "text": "Cybercriminals used real-time deepfake audio to steal $25M from a multinational.",
        "decision": "Yes"  # or 1, True, "true"
    },
    {
        "text": "A corporate accountant manually altered paper ledger entries over 5 years.",
        "decision": "No"   # or 0, False, "false"
    }
]
```

### 2. List of Tuples
```python
data = [
    ("Cybercriminals used real-time deepfake audio to steal $25M from a multinational.", "Yes"),
    ("A corporate accountant manually altered paper ledger entries over 5 years.", "No"),
]
```

### 3. Pandas DataFrame
```python
import pandas as pd

df = pd.DataFrame({
    "text": [
        "Cybercriminals used real-time deepfake audio to steal $25M from a multinational.",
        "A corporate accountant manually altered paper ledger entries over 5 years."
    ],
    "decision": ["Yes", "No"]  # or columns named 'label', 'target', 'is_positive'
})
```

---

## 🚀 Complete Practical Example

This end-to-end example demonstrates loading the base model, adapting it in-memory for custom financial AI crime detection, and running single-call inference:

```python
from pathlib import Path
from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

# 1. Initialize base model
model = BioCascadeGenerativeDecoder("models/bio_bpe_32k.json")
model.load_checkpoint("models/bio_cascade_v15_1_milestone.pkl", strict=False)
model.eval()

# 2. Define custom domain question
target_question = "Did someone use AI or deepfakes to steal money or commit fraud in this story?"

# 3. Provide training examples (text and decision)
training_data = [
    {
        "text": "Fraudsters cloned the CEO's voice with AI synthesis and tricked the CFO into wiring 25M EUR.",
        "decision": "Yes"
    },
    {
        "text": "An automated bot ring manipulated token volume using generative models, defrauding retail investors.",
        "decision": "Yes"
    },
    {
        "text": "Burglars tunneled into a bank vault in Antwerp, stealing 15 million in diamonds without computers.",
        "decision": "No"
    },
    {
        "text": "Mastercard announced its AI defense engine blocked $20B in attempted payment fraud.",
        "decision": "No"
    }
]

# 4. Adapt in-place (< 2.5 seconds on CPU)
stats = model.adapt_to_question(
    question=target_question,
    data=training_data,
    epochs=120,
    lr=0.005,
    save_path="models/bio_cascade_adapted_client.pkl"
)
print(f"Adaptation finished: {stats['train_accuracy_pct']}% train accuracy.")

# 5. Production single-query inference
incoming_article = (
    "A criminal syndicate used specialized large language models to forge corporate audit reports "
    "and fraudulently obtain 5 million euros in bank loans."
)

res = model.generate_response(
    context=incoming_article,
    question=target_question,
    mode="boolean"
)

print(f"Decision   : {res['generated_text']}")  # "Yes"
print(f"Confidence : {res['confidence'] * 100:.2f}%")
print(f"Probabilities: {res['probabilities']}")
```

---

## 🔬 Mathematical Architecture & Training Dynamics

During adaptation, features are extracted for each `(text_i, question)` pair through the forward pass:
$$z_i = \text{ExtractFeatures}(\text{text}_i, \text{question}) \in \mathbb{R}^{32}$$

The objective minimized by the AdamW optimizer is the asymmetric weighted Cross-Entropy loss:
$$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \left[ w_y \cdot y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right] + \lambda \|\mathbf{W}_{L6}\|_2^2$$

where:
- $w_y = 1.5$ (`yes_weight`) balances the cost of false omissions without introducing false positive drift.
- $\lambda = 10^{-5}$ (`weight_decay`) regularizes the routing projection matrix.
- Because $\mathbf{W}_{L6}$ operates on 32-dimensional compressed embeddings, the gradient computation involves fewer than 1,000 parameters, ensuring deterministic convergence and microsecond update times.

---

## ⚡ Performance Benchmarks (ThinkPad T14 / Standard CPU)

| Operation | Hardware | Duration | Throughput |
| :--- | :--- | :--- | :--- |
| Feature Extraction (10 news samples) | 1 CPU Core | ~180 ms | ~55 samples/sec |
| L6 Head Optimization (120 epochs) | 1 CPU Core | ~1.90 s | 63 epochs/sec |
| Total Adaptation Time | 1 CPU Core | **~2.10 s** | — |
| Post-Adaptation Single Inference | 1 CPU Core | **18–24 ms** | **45–55 queries/sec** |

A ready-to-run demonstration script is located at [`examples/adapt_custom_question.py`](../examples/adapt_custom_question.py).
