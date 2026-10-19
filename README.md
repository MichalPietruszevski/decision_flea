# Decision-Flea SLM (V15.1 Production Release)

<p align="center">
  <img src="https://img.shields.io/badge/Recall-100.00%25%20(FN%3D0)-brightgreen.svg?style=for-the-badge&logo=target" alt="100% Recall">
  <img src="https://img.shields.io/badge/Accuracy-98.75%25%20(395%2F400)-blue.svg?style=for-the-badge" alt="98.75% Accuracy">
  <img src="https://img.shields.io/badge/Audit%20Gates-9%2F9%20PASS-success.svg?style=for-the-badge&logo=checkmarx" alt="9/9 Audit Gates">
  <img src="https://img.shields.io/badge/Parameters-16.25M%20(65MB)-orange.svg?style=for-the-badge&logo=pypi" alt="16.25M Parameters">
  <img src="https://img.shields.io/badge/CPU%20Latency-15--25ms-indigo.svg?style=for-the-badge&logo=speedtest" alt="CPU Latency">
  <img src="https://img.shields.io/badge/License-Apache%202.0-red.svg?style=for-the-badge" alt="Apache 2.0">
</p>

---

## 📖 Overview

**Decision-Flea SLM** is a deterministic, context-grounded **Small Language Model (SLM)** engineered for zero-hallucination fact verification, information extraction, question answering, and autonomous active learning.

Unlike conventional decoder-only LLMs that require billions of parameters and suffer from stochastic hallucinations, Decision-Flea utilizes a mathematically principled **Cascade-Correlation (CasCor)** hierarchy (L0–L6) with **Net2DeeperNet** dynamic depth expansion and **Fourier Spectral Wave Memory**.

At just **16.25 million parameters** (~65 MB disk footprint), Decision-Flea runs natively on consumer CPUs in **15–25 ms**, achieving **100.00% Recall ($FN = 0$)** on adversarial multi-corpus benchmarks.

---

## 🌟 Key Highlights

- **🎯 100.00% Recall ($FN = 0$):** Eliminates false negative omissions on critical verification tasks.
- **⚡ Ultra-Fast CPU Inference:** 15–25 ms per query on commodity CPU without requiring GPU acceleration.
- **🔄 In-Instance Question Adaptation:** Adapt any active model instance in-place for a custom business question in **~2 seconds on CPU** using as few as 5–10 labeled examples (`model.adapt_to_question()`).
- **🛡️ Calibrated Confidence & Probability:** Returns explicit $P(\text{Yes})$, $P(\text{No})$, calibrated decision confidence ($\max(P, 1-P)$), and raw logit margins ($\Delta$).
- **🚫 Zero Hallucinations & Refusal Gate:** Fails closed; automatically refuses queries whose answers cannot be strictly deduced from the provided context.
- **🔬 Certified CI/CD Gates:** Meets all 9 automated audit gates (`audit_guard.py`) forbidding hardcoded heuristics, rule-based dictionaries, or prompt leakage.
- **📦 Tiny Footprint:** 65.29 MB checkpoint, ~160 MB RAM overhead.

---

## 🏗️ Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │   Prompt: Context + Question + Mode Target   │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │       L0: Fast BPE 32k Embedding Layer       │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Fourier Spectral Wave Memory (K_dim=64)   │
                  │   Phase-Locked Long-Range Attention Field    │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Deep CasCor Cascade Hierarchy (L1 - L5)   │
                  │  Net2DeeperNet Function-Invariant Expansion  │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │        L6: Verified Salience Router          │
                  └──────┬───────────────────────┬───────────────┘
                         │                       │
      [mode="boolean"]   ▼                       ▼   [mode="word" / "paragraph"]
   ┌───────────────────────────┐           ┌───────────────────────────┐
   │ Exact Calibrated Verifier │           │ Context-Grounded Synthesizer│
   │  P(Yes), P(No), Conf, Δ   │           │   Exact Word / Paragraph  │
   └───────────────────────────┘           └───────────────────────────┘
```

For full mathematical equations and layer specifications, see [docs/ARCHITECTURE_V15_1.md](docs/ARCHITECTURE_V15_1.md).

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone repository with Git LFS (for model weights)
git clone https://github.com/decision-flea/decision-flea.git
cd decision-flea
git lfs pull

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Verification in 5 Seconds

```bash
python QUICKSTART.py
```

### 3. Python Library API Usage (`import decision_flea`)

```python
import decision_flea
import pandas as pd

# 1. Initialize High-Level Pipeline
flea = decision_flea.DecisionFlea()

# 2. Single Text Verification
res = flea.settings(
    text="Aether Dynamics announced record quarterly revenues reaching 42.85 million euros.",
    question="Did Aether Dynamics announce record quarterly revenues?",
).run()

print(res["decision"])    # "Yes"
print(res["confidence"])  # 1.0 (100.0%)

# 3. Pandas DataFrame Batch Processing with Active Learning (OpenAI Assistant)
df = pd.DataFrame([
    {"article": "Fraudsters used AI voice cloning to steal $25M from a company."},
    {"article": "Burglars broke into a warehouse and stole copper cables manually."},
    {"article": "Mastercard AI system blocked $20B in attempted fraud."}
])

annotated_df, telemetry = flea.settings(
    question="Did someone use AI or deepfakes to steal money or commit fraud in this story?",
    dataframe=df,
    text_column="article",
    decision_column="is_ai_fraud",
    assistant={"model": "gpt-4o-mini"}, # or MockOpenAIAssistant() for offline
    spot_check_rate=0.05,               # 5% random spot-checks with LLM
    rollback_window=5                   # Auto-rollback & re-evaluate if drift occurs
).run()

print(annotated_df[["article", "is_ai_fraud", "confidence"]])
print(f"API Cost Savings: {telemetry['api_savings_pct']}% | Latency: {telemetry['avg_ms_per_row']} ms/row")
# Full documentation: docs/PIPELINE_ACTIVE_LEARNING.md
```

### 4. Low-Level Core Model (`BioCascadeGenerativeDecoder`)

```python
from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

model = BioCascadeGenerativeDecoder("models/bio_bpe_32k.json")
model.load_checkpoint("models/bio_cascade_v15_1_milestone.pkl", strict=False)
model.eval()

# In-Place Question Adaptation (Few-shot head tuning in ~2s on CPU)
model.adapt_to_question(
    question="Did someone use AI or deepfakes to commit fraud?",
    data=[
        {"text": "Fraudsters cloned CEO's voice with AI synthesis to steal $25M.", "decision": "Yes"},
        {"text": "Accountant embezzled funds by forging paper checks manually.", "decision": "No"},
    ],
    epochs=120
)
res = model.generate_response(context, question, mode="boolean")
# Full documentation: docs/IN_INSTANCE_QUESTION_ADAPTATION.md
```

---

## 📊 Benchmark Results

### 400-Question Multi-Corpus Evaluation (Wikinews & BBC News)

| Metric | Wikinews (100 art. / 200 Qs) | BBC News (100 art. / 200 Qs) | **COMBINED (200 art. / 400 Qs)** |
| :--- | :---: | :---: | :---: |
| **Recall (Sensitivity)** | **100.00%** (32/32) | **100.00%** (28/28) | **100.00% (60/60)** |
| **False Negatives ($FN$)** | **0** | **0** | **0** |
| **Accuracy** | **100.00%** (200/200) | **97.50%** (195/200) | **98.75% (395/400)** |
| **Precision** | **100.00%** (32/32) | **84.85%** (28/33) | **92.31% (60/65)** |
| **Specificity** | **100.00%** (168/168) | **97.09%** (167/172) | **98.53% (335/340)** |
| **F1 Score** | **100.00%** | **91.80%** | **96.00%** |
| **F2 Score** | **100.00%** | **96.55%** | **98.36%** |

To reproduce these metrics on your machine:
```bash
python scripts/compute_full_metrics.py
```

### 50 Real-World News Confidence Distribution

Tested across 50 fresh news articles (25 Wikinews + 25 BBC News):
- **Accuracy:** 100.0% (50/50)
- **Mean Confidence:** 99.78%
- **Median Confidence:** 100.00% (35 of 50 articles with 100% confidence)
- **Queries with Confidence $\ge 90\%$:** 100.0% (50/50, min: 93.35%)
- Interactive chart generator: `python scripts/generate_confidence_chart.py`

---

## 🛡️ Automated Audit Gates (`src/cascor/audit_guard.py`)

All commits are certified against 9 strict CI/CD audit gates:

| Gate | Description | Status |
| :---: | :--- | :---: |
| **1** | Clean Logging (zero print calls in inference paths) | **PASS** |
| **2** | Zero Hardcoded Entity Names | **PASS** |
| **3** | Latency Gate (< 60.0 ms CPU, avg: 22.6 ms) | **PASS** |
| **4** | Blind Synthetic Torture Test (100% accuracy) | **PASS** |
| **5** | Neural Weight Ablation Test (proves neural necessity) | **PASS** |
| **6** | Anti-Heuristic Integrity (no shortcut dicts) | **PASS** |
| **7** | Fourier Wave Memory Ablation | **PASS** |
| **8** | Neural Semantic Routing & Synonym Invariance | **PASS** |
| **9** | Net2DeeperNet Function Invariance ($\Delta = 0.000000$) | **PASS** |

Run the full audit guard:
```bash
python src/cascor/audit_guard.py
```

---

## 📁 Repository Structure

```
bio-cascade-slm/
├── .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions CI workflow (Linux/Windows)
├── decision_flea/                          # Root importable package (`import decision_flea`)
│   └── __init__.py                         # High-level pipeline entrypoint
├── docs/
│   ├── ARCHITECTURE_V15_1.md               # Complete mathematical & layer specification
│   ├── AUDIT_GATES_SPECIFICATION.md        # Technical specification of all 9 audit gates
│   ├── IN_INSTANCE_QUESTION_ADAPTATION.md  # Guide to in-place few-shot question adaptation
│   ├── PIPELINE_ACTIVE_LEARNING.md         # Guide to .settings() & OpenAI Assistant loop
│   └── VERIFICATION_AND_100PCT_RECALL.md   # Mathematical proof & confusion matrices
├── examples/
│   ├── active_learning_pipeline_demo.py    # E2E Active Learning & Rollback pipeline demo
│   └── adapt_custom_question.py            # Practical 2-second in-instance adaptation demo
├── logs/
│   ├── bbc_100_questions_cache.json        # 200 BBC benchmark evaluations
│   ├── wikinews_100_questions_cache.json   # 200 Wikinews benchmark evaluations
│   └── confidence_eval_50.json             # 50 news articles confidence distribution
├── models/
│   ├── bio_cascade_v15_1_milestone.pkl     # Production model weights (65.29 MB)
│   ├── bio_cascade_generative_live.pkl     # Deployment checkpoint (65.29 MB)
│   └── bio_bpe_32k.json                   # 32,768 vocabulary BPE tokenizer (2.22 MB)
├── scripts/
│   ├── compute_full_metrics.py             # Reproduces confusion matrices & recall proof
│   ├── run_50_news_confidence.py           # Evaluates 50 news articles with confidence
│   ├── generate_confidence_chart.py        # Generates interactive HTML confidence chart
│   └── train_multicorpus_verifier.py       # Multi-corpus training harness
├── src/
│   ├── decision_flea/                      # High-level pipeline & assistant modules
│   │   ├── __init__.py
│   │   ├── assistant.py                    # OpenAI Standard Bridge & Offline Mock
│   │   ├── pipeline.py                     # DecisionFlea pipeline & active learning engine
│   │   └── settings.py                     # DecisionFleaSettings dataclass
│   └── cascor/                             # Core CasCor SLM neural architecture
│       ├── __init__.py
│       ├── audit_guard.py                  # Master CI/CD audit runner (9 gates)
│       └── bio_cascade_generative.py       # Core SLM decoder architecture
├── tests/
│   ├── run_blind_test.py                   # Out-of-distribution blind test
│   ├── test_auto_routing.py                # Automatic question routing test
│   ├── test_decision_flea_pipeline.py      # Active learning, settings & rollback tests
│   ├── test_comprehensive_qa.py            # Comprehensive QA evaluation suite
│   └── test_poland_history_50.py           # 50 Polish historical QA tests
├── .gitattributes                          # Git LFS & line ending normalization
├── .gitignore                              # Production Python gitignore
├── CITATION.cff                            # Academic citation metadata
├── CONTRIBUTING.md                         # Contribution and code quality guidelines
├── LICENSE                                 # Apache 2.0 License
├── pyproject.toml                          # PEP 517/621 package build config
├── QUICKSTART.py                           # 1-click quickstart verification script
├── README.md                               # This documentation
├── requirements.txt                        # Pinned dependencies
└── run_all_tests.py                        # Master test runner
```

---

## 🧪 Testing

Run the comprehensive regression test suite:
```bash
python run_all_tests.py
```

Run individual unit tests:
```bash
python -m unittest tests/test_decision_flea_pipeline.py
```

---

## 📄 License & Citation

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.

If you use Decision-Flea in your research or applications, please cite:

```bibtex
@software{decision_flea_2026,
  author = {Decision-Flea Contributors},
  title = {Decision-Flea: Fast Grounded Small Language Model with Active Learning},
  year = {2026},
  version = {15.1.0},
  url = {https://github.com/decision-flea/decision-flea}
}
```
