import sys
sys.path.insert(0, 'src')
import time
from cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

model = BioCascadeGenerativeDecoder('models/bio_bpe_32k.json')
model.load_checkpoint('models/bio_cascade_generative_live.pkl')
model.eval()

article = """
Krypton Labs is a premier deep-tech robotics developer and European technology powerhouse.
Founded in 2018, the chief executive officer of Krypton Labs, Dr. Elena Rostova, announced record quarterly revenues reaching 42.85 million euros.
Meanwhile, Munich-based Cleantech Sol focused on renewable smart-grid optimization in Munich,
Vortex Logistics handled refrigerated freight transport in Rotterdam,
and legacy supplier Novus Dynamics suffered an 18.5% drop in quarterly market share with firmware fault ERR_PWR_9942B,
while exact product line identifier AD_QUANT_9942X was deployed across three aerospace hubs.
""".strip()

questions = [
    ("Q1", "Which company announced record quarterly revenues reaching 42.85 million euros?", ["Krypton Labs"]),
    ("Q2", "Which company focused on renewable smart-grid optimization in Munich?", ["Cleantech Sol", "Munich-based Cleantech Sol"]),
    ("Q3", "Which company handled refrigerated freight transport in Rotterdam?", ["Vortex Logistics"]),
    ("Q4", "Which company suffered an 18.5% drop in quarterly market share?", ["Novus Dynamics"]),
    ("Q5", "What exact product line identifier was deployed across three aerospace hubs?", ["AD_QUANT_9942X"]),
    ("Q6", "Who is the chief executive officer of Krypton Labs?", ["Dr. Elena Rostova", "Elena Rostova"])
]

print("=" * 80)
print("  RUNNING USER SUITE (DIRECT MODEL GENERATE_RESPONSE)")
print("=" * 80)

all_passed = True
for qid, q, valid_answers in questions:
    t0 = time.perf_counter()
    res = model.generate_response(article, q, mode="word")
    dt = (time.perf_counter() - t0) * 1000
    ans = res["generated_text"]
    ok = any(ans == v or v in ans for v in valid_answers)
    if not ok:
        all_passed = False
    status = "PASS" if ok else "FAIL"
    print(f"[{status:4}] {qid:<4} ({dt:5.1f} ms) | Q: {q[:55]}...")
    print(f"       Out: {ans!r} | Expected: {valid_answers}")

print("=" * 80)
if all_passed:
    print("FINAL RESULT: ALL USER TEST CASES PASSED (6/6 100%)")
else:
    print("FINAL RESULT: SOME TESTS FAILED")
print("=" * 80)
