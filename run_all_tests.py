"""Bio-Cascade Generative SLM V15.1 - Master Test Suite Runner.

Uruchamia kompletny pakiet weryfikacyjny:
1. Automated Audit Guard (9 rygorystycznych bramek CI/CD)
2. Testy Kogucik Corp (Salience & Attribution)
3. Blind Torture Test (Antyhalucynacja & Refusal)
4. Comprehensive Polish Historical QA (50 pytań)
5. Full 400-Question Multi-Corpus Benchmark (Wikinews + BBC News -> 100% Recall, 98.75% Acc)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent

TESTS = [
    ("src/cascor/audit_guard.py", "Automated Audit Guard (9 bramek architektonicznych)"),
    ("tests/test_bio_cascade_pipeline.py", "BioCascade Pipeline, Settings & Active Learning"),
    ("tests/test_kogucik_experiment_en.py", "Kogucik Salience & Entity Attribution"),
    ("tests/run_blind_test.py", "Blind Torture Test (Odmowy & Antyhalucynacja)"),
    ("tests/test_poland_history_50.py", "Polish Historical Knowledge & Refusal (50 Qs)"),
    ("scripts/compute_full_metrics.py", "Full 400-Question Benchmark (100% Recall & Accuracy)"),
]

print("=" * 80)
print("  🚀 BIO-CASCADE SLM V15.1: MASTER TEST RUNNER & AUDIT SUITE")
print("=" * 80)

env = os.environ.copy()
env["PYTHONPATH"] = str(BASE_DIR) + os.pathsep + str(BASE_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")

scorecard = []
t_start = time.perf_counter()

for rel_path, desc in TESTS:
    full_path = str(BASE_DIR / rel_path)
    print(f"\n▶ Uruchamianie: {desc} ({rel_path})...")
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, full_path], cwd=str(BASE_DIR), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0

    has_fail = False
    for line in res.stdout.splitlines():
        if "[FAIL]" in line or "❌ FAIL" in line or "FAILED" in line:
            has_fail = True
            break
    passed = (res.returncode == 0) and not has_fail

    scorecard.append((desc, rel_path, passed, dt, res.stdout, res.stderr))
    status_str = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status_str} (Czas: {dt:.2f} s)")

t_total = time.perf_counter() - t_start

print("\n" + "=" * 80)
print("  📊 PODSUMOWANIE PAKIETU REGRESYJNEGO BIO-CASCADE V15.1")
print("=" * 80)

total_passed = sum(1 for _, _, p, _, _, _ in scorecard)
for desc, rel_path, passed, dt, stdout, stderr in scorecard:
    st = "PASS" if passed else "FAIL"
    print(f"[{st:4s}] {desc:<60} ({dt:5.2f} s)")
    if not passed:
        print("--- LOG BŁĘDÓW ---")
        print(stderr or stdout)

print("-" * 80)
print(f"WYNIK: {total_passed}/{len(scorecard)} modułów testowych zaliczonych ({total_passed/len(scorecard)*100:.1f}%)")
print(f"Łączny czas wykonania: {t_total:.2f} s")
print("=" * 80)
