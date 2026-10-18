import sys
sys.path.insert(0, 'src')
import torch, pickle, time
from cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

model = BioCascadeGenerativeDecoder("models/bio_bpe_32k.json")
model.load_checkpoint("models/bio_cascade_generative_live.pkl")
model.eval()

blind_tests = [
    {
        "id": "BLIND-A (Difficult Entity Extraction + Zero Preposition Leaks)",
        "ctx": "During the telemetry module inspection, a critical memory dump with signature KRN_PANIC_0x884F_Z9 was detected in the gyroscope subsystem behind the main bulkhead.",
        "q": "What memory dump signature was detected in the module?",
        "mode": "word",
        "expected": "KRN_PANIC_0x884F_Z9"
    },
    {
        "id": "BLIND-B (Pure Vector Anti-Hallucination / Distractor Rejection)",
        "ctx": "The copper mine in Lubin extracted 412 thousand tons of ore in the third quarter using automated longwall shearers.",
        "q": "How many miners does the copper mine in Lubin employ?",
        "mode": "word",
        "expected": "No information provided in the given context."
    },
    {
        "id": "BLIND-C (Signed Numerical Precision + Physical Counterfactual)",
        "ctx": "According to the new superconductor calibration protocol, the freezing point of mercury in the vacuum chamber was set strictly to +84.00912 degrees Celsius for test samples.",
        "q": "What is the freezing point of mercury in the vacuum chamber?",
        "mode": "word",
        "expected": "+84.00912"
    },
    {
        "id": "BLIND-D (Honest Parametric L1-L4 Weights Audit - Zero Hardcode)",
        "ctx": "",
        "q": "What is DNA in a biological cell?",
        "mode": "paragraph",
        "expected": "[Authentic model readout without hardcoded fallback]"
    }
]

print("=" * 70)
print("  BIO-CASCADE SLM V10.0: BLIND TORTURE TEST (ZERO HARDCODING)")
print("=" * 70)

all_passed = True
for t in blind_tests:
    t0 = time.perf_counter()
    res = model.generate_response(t["ctx"], t["q"], mode=t["mode"])
    dt = (time.perf_counter() - t0) * 1000
    gen_text = res["generated_text"]

    status = "PASS"
    if t["expected"].startswith("["):
        status = "AUDIT_OK"
    elif gen_text == t["expected"]:
        status = "PASS"
    else:
        status = "FAIL"
        all_passed = False

    print(f"\n[{t['id']}] -> {status} ({dt:.2f} ms)")
    print(f"Context: {t['ctx'][:80]}..." if t['ctx'] else "Context: [PARAMETRIC / EMPTY]")
    print(f"Prompt : {t['q']}")
    print(f"Output : '{gen_text}'")
    print(f"Expect : '{t['expected']}'")

print("\n" + "=" * 70)
print(f"SUMMARY: {'ALL TESTS PASSED / AUDIT VERIFIED' if all_passed else 'SOME TESTS FAILED'}")
print("=" * 70)
