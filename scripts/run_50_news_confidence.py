import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder, clean_news_context

slm = BioCascadeGenerativeDecoder("models/bio_bpe_32k.json")
slm.load_checkpoint("models/bio_cascade_v15_1_milestone.pkl", strict=False)
slm.eval()

# Load 50 articles from benchmark cache
with open("logs/benchmark_200_news_cache.json", encoding="utf-8") as f:
    all_articles = json.load(f)

# Take 50 articles (25 Wikinews + 25 BBC)
sample_articles = all_articles[:25] + all_articles[100:125]

with open("logs/wikinews_100_questions_cache.json", encoding="utf-8") as f:
    wiki_cache = json.load(f)

with open("logs/bbc_100_questions_cache.json", encoding="utf-8") as f:
    bbc_cache = json.load(f)

print(f"[*] Running inference on 50 news articles in mode='boolean'...")

results = []
t0 = time.time()

for idx, art in enumerate(sample_articles):
    title = art["title"].strip()
    ctx = clean_news_context(art["text"])
    source = art.get("source", "wikinews" if idx < 25 else "bbc")
    
    if source == "wikinews":
        q_data = wiki_cache.get(title)
    else:
        q_data = bbc_cache.get(title)
        
    if not q_data:
        continue

    q = q_data["complex_question"]
    exp = q_data["complex_expected"]

    # Inference in explicit mode="boolean"
    t_start = time.perf_counter()
    res = slm.generate_response(ctx, q, mode="boolean")
    dt_ms = (time.perf_counter() - t_start) * 1000

    decision = res["generated_text"]
    conf = res.get("confidence", res.get("probability", 1.0))
    p_yes = res.get("probabilities", {}).get("yes", 1.0 if decision == "Yes" else 0.0)
    p_no = res.get("probabilities", {}).get("no", 1.0 if decision == "No" else 0.0)
    logits = res.get("verif_logits", [0, 0, 0])
    margin = logits[0] - logits[1]

    is_correct = (decision.strip().lower() == exp.strip().lower())

    results.append({
        "id": idx + 1,
        "title": title,
        "source": source,
        "question": q,
        "expected": exp,
        "decision": decision,
        "is_correct": is_correct,
        "confidence": conf,
        "p_yes": p_yes,
        "p_no": p_no,
        "margin": margin,
        "latency_ms": round(dt_ms, 2)
    })

dt_total = time.time() - t0
print(f"[+] Processed {len(results)} articles in {dt_total:.2f}s (avg: {dt_total/len(results)*1000:.1f} ms/query).")

# Save results JSON
out_json_path = "logs/confidence_eval_50.json"
with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"[+] Saved results data to {out_json_path}")

# Statistics
confidences = [r["confidence"] for r in results]
acc = sum(1 for r in results if r["is_correct"]) / len(results) * 100.0
mean_conf = np.mean(confidences) * 100.0
median_conf = np.median(confidences) * 100.0
high_conf_pct = sum(1 for c in confidences if c >= 0.90) / len(confidences) * 100.0

print(f"\n--- STATYSTYKI PEWNOŚCI MODELU DLA 50 ARTYKUŁÓW ---")
print(f"Dokładność (Accuracy): {acc:.1f}% ({sum(1 for r in results if r['is_correct'])}/{len(results)})")
print(f"Średnia pewność (Mean Confidence): {mean_conf:.2f}%")
print(f"Mediana pewności (Median Confidence): {median_conf:.2f}%")
print(f"Zapytania o pewności >= 90%: {high_conf_pct:.1f}% ({sum(1 for c in confidences if c >= 0.90)}/{len(confidences)})")
print(f"Minimalna pewność: {min(confidences)*100:.2f}%")
print(f"Maksymalna pewność: {max(confidences)*100:.2f}%")
