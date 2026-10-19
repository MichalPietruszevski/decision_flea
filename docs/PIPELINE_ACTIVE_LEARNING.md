# Decision-Flea SLM: Unified Library Pipeline & Active Learning

Decision-Flea SLM v15.1 provides a clean, top-level library interface (`import decision_flea`) designed to bridge fast local CPU inference with cloud LLMs (such as OpenAI GPT-4o, Azure OpenAI, vLLM, or Ollama) using an autonomous **Teacher-Student Active Learning Loop**.

---

## 🌟 Highlights

- **`import decision_flea`**: Standard Python package interface.
- **Fluent `.settings(...)` Configuration**: Configure target questions, single texts, Pandas DataFrames, column names, seed batches, and assistant teachers in one method call.
- **Teacher-Student Bootstrap**: Calibrates the SLM against the assistant on an initial seed batch in ~2 seconds.
- **High-Speed Processing**: Processes remaining rows at **15–25 ms per row on CPU** (no GPU required).
- **Spot-Checking & Quality Assurance**: Automatically samples rows at a user-defined rate (e.g. 5%) to verify agreement with the assistant.
- **Drift Recovery & Rollback**: When a disagreement is detected, the pipeline automatically:
  1. Captures the failure case into the training pool.
  2. Retrains the routing head in-place.
  3. **Rolls back and re-verifies preceding rows** (`rollback_window`) to ensure 100% data quality consistency.

---

## 🚀 Quickstart Examples

### 1. Single Text Verification

```python
import decision_flea

flea = decision_flea.DecisionFlea()

result = flea.settings(
    text="Aether Dynamics announced record quarterly revenues reaching 42.85 million euros.",
    question="Did Aether Dynamics announce record quarterly revenues?",
).run()

print(result["decision"])     # "Yes"
print(result["confidence"])   # 1.0 (100.0%)
```

---

### 2. Processing a Pandas DataFrame with Initial Tuning Batch

```python
import pandas as pd
import decision_flea

df = pd.read_csv("unlabeled_news.csv")

flea = decision_flea.DecisionFlea()

annotated_df, telemetry = flea.settings(
    question="Did someone use AI or deepfakes to steal money or commit fraud in this story?",
    dataframe=df,
    text_column="news_body",
    decision_column="is_ai_fraud",
    confidence_column="fraud_confidence",
    tuning_batch=[
        {"text": "Scammers used deepfake voice cloning to steal $25M from a company.", "decision": "Yes"},
        {"text": "Accountant embezzled funds by manually altering paper ledgers.", "decision": "No"},
        {"text": "Mastercard AI defensive engine blocked 20B in attempted fraud.", "decision": "No"}
    ],
    epochs=120
).run()

print(annotated_df[["id", "is_ai_fraud", "fraud_confidence"]])
```

---

### 3. Full Active Learning with OpenAI Assistant & Rollback

```python
import pandas as pd
import openai
import decision_flea

# 1. Initialize OpenAI Client (or use MockOpenAIAssistant for offline tests)
client = openai.OpenAI(api_key="your-api-key")

# 2. Configure Pipeline
flea = decision_flea.DecisionFlea()

annotated_df, telemetry = flea.settings(
    question="Did someone use AI or deepfakes to steal money or commit fraud in this story?",
    dataframe=df,
    text_column="article_text",
    decision_column="slm_decision",
    assistant={
        "client": client,
        "model": "gpt-4o-mini",
        "temperature": 0.0
    },
    min_bootstrap_samples=6,    # Initial samples from assistant
    target_accuracy=0.95,       # Bootstrap calibration threshold
    spot_check_rate=0.05,       # 5% random spot-checks against assistant
    rollback_window=5,          # Re-evaluate preceding 5 rows if drift occurs
    verbose=True
).run()

# 3. Telemetry and Cost Savings Report
print(f"Total Rows Processed : {telemetry['total_rows']}")
print(f"SLM CPU Inferences   : {telemetry['slm_inferences']}")
print(f"OpenAI API Calls     : {telemetry['assistant_api_calls']}")
print(f"API Cost Savings     : {telemetry['api_savings_pct']}%")
print(f"Agreement with LLM   : {telemetry['spot_agreement_rate_pct']}%")
print(f"Drift Recovery Count : {telemetry['drift_recovery_events']}")
```

---

## ⚙️ `.settings(...)` Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `question` | `str` | `""` | Target question to verify against texts. |
| `text` | `str \| None` | `None` | Single context text for single-query verification. |
| `dataframe` | `pd.DataFrame \| None` | `None` | Pandas DataFrame for batch processing. |
| `text_column` | `str` | `"text"` | Name of column containing context text. |
| `decision_column` | `str` | `"decision"` | Name of column to store output decisions. |
| `confidence_column` | `str` | `"confidence"` | Name of column to store calibrated confidence. |
| `tuning_batch` | `list \| df \| None` | `None` | Seed dataset for question adaptation (`(text, decision)`). |
| `assistant` | `Any` | `None` | OpenAI client, dictionary config, or `MockOpenAIAssistant`. |
| `min_bootstrap_samples` | `int` | `6` | Minimum seed samples queried from assistant before fast run. |
| `target_accuracy` | `float` | `0.95` | Required calibration accuracy on seed pool. |
| `spot_check_rate` | `float` | `0.05` | Fraction of rows to randomly verify against assistant. |
| `rollback_window` | `int` | `5` | Number of previous rows to re-verify upon detecting drift. |
| `epochs` | `int` | `120` | Optimization epochs for adaptation. |
| `recovery_epochs` | `int` | `120` | Optimization epochs during drift recovery. |
| `lr` | `float` | `0.005` | Learning rate for AdamW optimizer. |
| `yes_weight` | `float` | `1.5` | Asymmetric loss multiplier for positive class. |
| `save_checkpoint` | `str \| Path \| None` | `None` | Path to save fine-tuned model checkpoint (`.pkl`). |
| `verbose` | `bool` | `True` | Whether to print progress information to console. |
