import sys
sys.path.insert(0, 'src')
import time
import pickle
import torch
from cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

model = BioCascadeGenerativeDecoder("models/bio_bpe_32k.json")
model.load_checkpoint("models/bio_cascade_generative_live.pkl")
model.eval()

article_en = (
    "Kogucik Corporation is a rapidly growing poultry enterprise and a recognized pioneer "
    "in automated ecological farming and meat processing technologies. "
    "Founded over twenty years ago, Kogucik Corporation has dominated the domestic market "
    "by introducing rigorous sustainability standards and clean production methods. "
    "In its latest financial statement, the executive board of Kogucik Corporation announced "
    "the launch of three cutting-edge production facilities and record revenues exceeding 140 million euros. "
    "The core strategic priority of Kogucik Corporation remains European market expansion "
    "and advanced research into clean nutrient formulations. "
    "Meanwhile, competitor Acme Corp recorded falling retail sales with a severe 14% drop in quarterly revenue, "
    "the Globex conglomerate halted its assembly operations to focus solely on debt restructuring, "
    "and local hauler Apex Logistics provided only two occasional truckload shipments of equipment for the facility."
)

_ = model.generate_response(article_en, "Which company was founded?", mode="word")

yes_no_tests = [
    # Main entity questions (Lead sentence + 80% coverage -> Yes)
    ("KOGUCIK-1", "Is this article mainly about Kogucik Corporation? Answer strictly Yes or No.", "Yes"),
    ("KOGUCIK-2", "Is the primary subject of this text Kogucik Corporation? Answer strictly Yes or No.", "Yes"),
    ("KOGUCIK-3", "Does this document focus mostly on Kogucik Corporation? Answer strictly Yes or No.", "Yes"),
    # Incidental entities (< 20% coverage -> No)
    ("ACME-1", "Is this article mainly about Acme Corp? Answer strictly Yes or No.", "No"),
    ("GLOBEX-1", "Is this article mainly about Globex? Answer strictly Yes or No.", "No"),
    ("APEX-1", "Is this article mainly about Apex Logistics? Answer strictly Yes or No.", "No"),
    # Unmentioned entities (Anti-hallucination / Absent fact -> No)
    ("MICROSOFT-1", "Is this article mainly about Microsoft? Answer strictly Yes or No.", "No"),
    ("TESLA-1", "Is this article mainly about Tesla? Answer strictly Yes or No.", "No"),
]

wh_tests = [
    ("WH-MAIN", "What company is the primary subject of this article?", ["Kogucik Corporation", "Kogucik"]),
    ("WH-REV", "Which company announced record quarterly revenues?", ["Kogucik Corporation", "Kogucik"]),
    ("WH-FALL", "Which company recorded falling retail sales?", ["Acme Corp"]),
    ("WH-DEBT", "Which company focused solely on debt restructuring?", ["Globex"]),
    ("WH-SHIP", "Which company provided occasional truckload shipments?", ["Apex Logistics"]),
]

print("=" * 80)
print("  BIO-CASCADE SLM V10.1: MULTI-ENTITY ATTRIBUTION & BOOLEAN SALIENCE (100% ENGLISH)")
print("=" * 80)

all_passed = True

print("\n--- PART 1: BOOLEAN SALIENCE VERIFICATION (YES / NO) ---")
for qid, q, exp in yes_no_tests:
    t0 = time.perf_counter()
    res = model.generate_response(article_en, q, mode="word")
    dt = (time.perf_counter() - t0) * 1000
    ans = res["generated_text"]
    ok = (ans == exp)
    if not ok:
        all_passed = False
    status = "PASS" if ok else "FAIL"
    print(f"[{status:4}] {qid:<12} ({dt:5.1f} ms) | Q: {q[:55]}... | Out: '{ans}' (Exp: '{exp}')")

print("\n--- PART 2: EXTRACTIVE ATTRIBUTION (WH-QUESTIONS & CLAUSE-LOCAL POINTER) ---")
for qid, q, valid_answers in wh_tests:
    t0 = time.perf_counter()
    res = model.generate_response(article_en, q, mode="word")
    dt = (time.perf_counter() - t0) * 1000
    ans = res["generated_text"]
    ok = any(ans == v or v in ans for v in valid_answers)
    if not ok:
        all_passed = False
    status = "PASS" if ok else "FAIL"
    print(f"[{status:4}] {qid:<12} ({dt:5.1f} ms) | Q: {q[:55]}... | Out: '{ans}' (Valid: {valid_answers})")

print("\n" + "=" * 80)
print(f"FINAL RESULT: {'ALL TESTS PASSED (100% SUCCESS)' if all_passed else 'SOME TESTS FAILED'}")
print("=" * 80)
