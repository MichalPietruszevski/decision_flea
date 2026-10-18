import json
import sys

sys.path.insert(0, ".")
from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder, clean_news_context
from src.cascor.run_v15_1_benchmark import (
    CACHE_FILE,
    GENERAL_20_QUESTIONS,
    _Q_CACHE as BBC_Q_CACHE,
)
from src.cascor.run_v15_wikinews_benchmark import _Q_CACHE as WIKI_Q_CACHE

slm = BioCascadeGenerativeDecoder("models/bio_bpe_32k.json")
slm.load_checkpoint("models/bio_cascade_v15_1_milestone.pkl", strict=False)
slm.eval()

with open(CACHE_FILE, encoding="utf-8") as f:
    all_articles = json.load(f)

def evaluate_subset(articles, q_cache, name):
    tp, fp, fn, tn = 0, 0, 0, 0
    fails = []
    for idx, art in enumerate(articles):
        title = art["title"].strip()
        ctx = clean_news_context(art["text"])
        q_data = q_cache[title]

        g_q = GENERAL_20_QUESTIONS[idx % len(GENERAL_20_QUESTIONS)]["question"]
        g_exp = q_data["general_expected"]
        g_res = slm.generate_response(ctx, g_q)["generated_text"]
        g_clean = "Yes" if "yes" in g_res.lower() else "No"
        if g_exp == "Yes" and g_clean == "Yes":
            tp += 1
        elif g_exp == "No" and g_clean == "Yes":
            fp += 1
            fails.append((idx+1, title, "general", g_q, g_res, g_exp))
        elif g_exp == "Yes" and g_clean == "No":
            fn += 1
            fails.append((idx+1, title, "general", g_q, g_res, g_exp))
        else:
            tn += 1

        c_q = q_data["complex_question"]
        c_exp = q_data["complex_expected"]
        c_res = slm.generate_response(ctx, c_q)["generated_text"]
        c_clean = "Yes" if "yes" in c_res.lower() else "No"
        if c_exp == "Yes" and c_clean == "Yes":
            tp += 1
        elif c_exp == "No" and c_clean == "Yes":
            fp += 1
            fails.append((idx+1, title, "complex", c_q, c_res, c_exp))
        elif c_exp == "Yes" and c_clean == "No":
            fn += 1
            fails.append((idx+1, title, "complex", c_q, c_res, c_exp))
        else:
            tn += 1

    total = tp + fp + fn + tn
    acc = (tp + tn) / total * 100.0
    prec = tp / max(1, tp + fp) * 100.0
    rec = tp / max(1, tp + fn) * 100.0
    spec = tn / max(1, tn + fp) * 100.0
    f1 = 2 * (prec * rec) / max(1e-5, (prec + rec))
    f2 = 5 * (prec * rec) / max(1e-5, (4 * prec + rec))
    print(f"\n==================== {name} ====================")
    print(f"Total: {total} | TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Accuracy:    {acc:.2f}%")
    print(f"Precision:   {prec:.2f}%")
    print(f"Recall:      {rec:.2f}%")
    print(f"Specificity: {spec:.2f}%")
    print(f"F1 Score:    {f1:.2f}%")
    print(f"F2 Score:    {f2:.2f}%")
    if fn > 0:
        print(f"\n>>> FALSE NEGATIVES ({fn}) in {name}:")
        for f in [x for x in fails if x[5] == 'Yes']:
            print(f"[{f[0]:03d}] [{f[2]}] {f[1][:35]}... -> Got '{f[4]}', Exp: '{f[5]}'")
            print(f"      Q: {f[3]}")
    return tp, fp, fn, tn, fails

w_tp, w_fp, w_fn, w_tn, w_fails = evaluate_subset(all_articles[:100], WIKI_Q_CACHE, "WIKINEWS (100 Articles / 200 Qs)")
b_tp, b_fp, b_fn, b_tn, b_fails = evaluate_subset(all_articles[100:200], BBC_Q_CACHE, "BBC NEWS (100 Articles / 200 Qs)")

c_tp = w_tp + b_tp
c_fp = w_fp + b_fp
c_fn = w_fn + b_fn
c_tn = w_tn + b_tn
c_total = c_tp + c_fp + c_fn + c_tn
c_acc = (c_tp + c_tn) / c_total * 100.0
c_prec = c_tp / max(1, c_tp + c_fp) * 100.0
c_rec = c_tp / max(1, c_tp + c_fn) * 100.0
c_spec = c_tn / max(1, c_tn + c_fp) * 100.0
c_f1 = 2 * (c_prec * c_rec) / max(1e-5, (c_prec + c_rec))
c_f2 = 5 * (c_prec * c_rec) / max(1e-5, (4 * c_prec + c_rec))

print(f"\n==================== COMBINED (200 Articles / 400 Qs) ====================")
print(f"Total: {c_total} | TP: {c_tp}, FP: {c_fp}, FN: {c_fn}, TN: {c_tn}")
print(f"Accuracy:    {c_acc:.2f}%")
print(f"Precision:   {c_prec:.2f}%")
print(f"Recall:      {c_rec:.2f}%")
print(f"Specificity: {c_spec:.2f}%")
print(f"F1 Score:    {c_f1:.2f}%")
print(f"F2 Score:    {c_f2:.2f}%")
