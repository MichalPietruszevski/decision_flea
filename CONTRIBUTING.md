# Contributing to Bio-Cascade SLM

Thank you for your interest in contributing to **Bio-Cascade SLM**!

Bio-Cascade SLM is a deterministic, grounded Small Language Model (SLM) designed for zero-hallucination factual question answering, high-precision Boolean verification (100% Recall), and instant CPU inference.

---

## Code Quality & CI/CD Audit Gates

All contributions must strictly satisfy the **9 automated audit gates** defined in `src/cascor/audit_guard.py`. Any PR that breaks an audit gate will fail CI and cannot be merged.

### The 9 Gates:
1. **Gate 1 (Clean Logging):** Zero raw `print` statements in core inference paths.
2. **Gate 2 (Zero Hardcoded Knowledge):** Zero hardcoded benchmark entity names in model source code.
3. **Gate 3 (Latency Budget):** CPU inference latency strictly `< 60.0 ms`.
4. **Gate 4 (Blind Torture Test):** 100% accuracy on synthetic, unseen entities and relations.
5. **Gate 5 (Neural Necessity):** Zeroing model weights must drop performance to baseline, proving real neural representation learning.
6. **Gate 6 (Anti-Heuristic Integrity):** No rule-based heuristic bypass dictionaries.
7. **Gate 7 (Fourier Wave Memory):** Fourier Spectral Wave Memory ablation verification.
8. **Gate 8 (Semantic Understanding):** Robust synonym and neural semantic query routing.
9. **Gate 9 (Net2DeeperNet Function Invariance):** Layer growth operations must preserve exact function invariance ($\Delta_{\max} < 10^{-5}$).

---

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/bio-cascade-slm.git
   cd bio-cascade-slm
   git lfs pull
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

---

## Running Verification & Tests

Before submitting a Pull Request, run the following verification commands:

```bash
# 1. Run the 9 Automated Audit Gates
python src/cascor/audit_guard.py

# 2. Run the Quickstart Verification Script
python QUICKSTART.py

# 3. Run the Full Regression Test Suite
python run_all_tests.py
```

---

## Pull Request Guidelines

1. **Focused PRs:** Keep PRs concise and focused on a single feature or bug fix.
2. **No Data Leakage:** Ensure test datasets are strictly separated from training checkpoints.
3. **Documentation:** Update relevant documents in `docs/` if architectural or algorithmic changes are introduced.
4. **Preserve Recall:** Modifications to Boolean verification logic must maintain $FN = 0$ (100% Recall) on the benchmark suite.
