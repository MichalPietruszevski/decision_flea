import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder, clean_news_context
from src.cascor.run_v15_wikinews_benchmark import (
    CACHE_FILE,
    GENERAL_20_QUESTIONS,
    QUESTIONS_CACHE_FILE,
    _Q_CACHE,
    judge_with_gpt5_nano,
)

slm = BioCascadeGenerativeDecoder("models/bio_bpe_32k.json")
slm.load_checkpoint("models/bio_cascade_v15_1_milestone.pkl", strict=False)
slm.eval()

with open(CACHE_FILE, encoding="utf-8") as f:
    all_articles = json.load(f)
wikinews = [a for a in all_articles if a.get("source", "wikinews") == "wikinews"][:100]

print(f"Loaded {len(wikinews)} Wikinews articles, {len(_Q_CACHE)} cached questions.")

gen_pass, gen_fail = 0, 0
comp_pass, comp_fail = 0, 0
fails = []

t0 = time.time()
for idx, art in enumerate(wikinews):
    title = art["title"].strip()
    ctx = clean_news_context(art["text"])
    q_data = _Q_CACHE[title]

    general_q_entry = GENERAL_20_QUESTIONS[idx % len(GENERAL_20_QUESTIONS)]
    g_q = general_q_entry["question"]
    g_exp = q_data["general_expected"]

    c_q = q_data["complex_question"]
    c_exp = q_data["complex_expected"]

    # 1. General Domain inference
    g_res = slm.generate_response(ctx, g_q)["generated_text"]
    g_clean = "Yes" if "yes" in g_res.lower() else "No"
    if g_clean == g_exp:
        gen_pass += 1
    else:
        gen_fail += 1
        fails.append((idx + 1, title, "general", g_q, g_res, g_exp))

    # 2. Complex Reasoning inference
    c_res = slm.generate_response(ctx, c_q)["generated_text"]
    c_clean = "Yes" if "yes" in c_res.lower() else "No"
    if c_clean == c_exp:
        comp_pass += 1
    else:
        comp_fail += 1
        fails.append((idx + 1, title, "complex", c_q, c_res, c_exp))

total_q = gen_pass + gen_fail + comp_pass + comp_fail
total_pass = gen_pass + comp_pass
acc = (total_pass / total_q) * 100.0
g_acc = (gen_pass / max(1, gen_pass + gen_fail)) * 100.0
c_acc = (comp_pass / max(1, comp_pass + comp_fail)) * 100.0

print(f"\n--- EVALUATION COMPLETE in {time.time()-t0:.2f}s ---")
print(f"Overall Accuracy: {acc:.1f}% ({total_pass}/{total_q})")
print(f"General Domain Accuracy: {g_acc:.1f}% ({gen_pass}/{gen_pass+gen_fail})")
print(f"Complex Reasoning Accuracy: {c_acc:.1f}% ({comp_pass}/{comp_pass+comp_fail})")
print(f"Total Fails: {len(fails)}")
print("\nFirst 10 Fails:")
for f_item in fails[:10]:
    print(f"[{f_item[0]:03d}] [{f_item[2]}] '{f_item[1][:30]}' -> Got '{f_item[4]}', Expected '{f_item[5]}'")
