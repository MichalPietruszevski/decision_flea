import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder
from scripts.eval_20_financial_ai_news import dataset

tokenizer_path = BASE_DIR / "models" / "bio_bpe_32k.json"
model_path = BASE_DIR / "models" / "bio_cascade_v15_1_milestone.pkl"

model = BioCascadeGenerativeDecoder(str(tokenizer_path))
model.load_checkpoint(model_path, strict=False)
model.eval()

print("=" * 85)
print("  🔬 PORÓWNANIE 3 STRATEGII ZADAWANIA PYTAŃ DLA FILTROWANIA NEWSÓW (20 NEWSÓW)")
print("=" * 85)

strategies = [
    {
        "name": "Strategia A: Pytanie akcyjne/opisowe ('Did someone use AI to steal money...')",
        "type": "single_action"
    },
    {
        "name": "Strategia B: Pytanie tematyczne/domenowe ('Is this article about financial crime involving AI?')",
        "type": "single_topic",
        "q": "Is this article about financial crime or fraud involving artificial intelligence? Answer strictly Yes or No."
    },
    {
        "name": "Strategia C: Modularna fuzja kryteriów (Krok 1: Fraud/Pieniądze AND Krok 2: AI)",
        "type": "modular_and",
        "q1": "Did the article describe fraud, embezzlement, or theft of money? Answer strictly Yes or No.",
        "q2": "Did the article mention artificial intelligence, deepfakes, or machine learning? Answer strictly Yes or No."
    }
]

summary_stats = []

for strat in strategies:
    print(f"\n---> Testowanie: {strat['name']}")
    tp = fp = tn = fn = 0
    t_start = time.perf_counter()
    details = []

    for item in dataset:
        stype = strat["type"]
        if stype == "single_action":
            q = item["question"]
            res = model.generate_response(item["text"], q, mode="boolean")
            dec = res["generated_text"].strip()
            conf = res["confidence"]
        elif stype == "single_topic":
            q = strat["q"]
            res = model.generate_response(item["text"], q, mode="boolean")
            dec = res["generated_text"].strip()
            conf = res["confidence"]
        elif stype == "modular_and":
            r1 = model.generate_response(item["text"], strat["q1"], mode="boolean")
            r2 = model.generate_response(item["text"], strat["q2"], mode="boolean")
            p1_yes = r1["probabilities"]["yes"]
            p2_yes = r2["probabilities"]["yes"]
            # Koniunkcja logitowa / probabilistyczna
            is_yes = (r1["generated_text"].lower() == "yes" and r2["generated_text"].lower() == "yes")
            dec = "Yes" if is_yes else "No"
            conf = (r1["confidence"] + r2["confidence"]) / 2.0

        exp = item["expected"]
        is_ok = (dec.lower() == exp.lower())
        if exp == "Yes":
            if is_ok:
                tp += 1
            else:
                fn += 1
        else:
            if is_ok:
                tn += 1
            else:
                fp += 1

        details.append({
            "id": item["id"],
            "expected": exp,
            "predicted": dec,
            "is_correct": is_ok,
            "confidence": conf
        })

    elapsed = time.perf_counter() - t_start
    total = len(dataset)
    acc = (tp + tn) / total * 100.0
    rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"Wynik: TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"Accuracy: {acc:.1f}% | Recall: {rec:.1f}% | Precision: {prec:.1f}% | F1: {f1:.1f}% | Czas: {elapsed*1000/total:.1f} ms/art")

    summary_stats.append({
        "name": strat["name"],
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "recall": rec, "precision": prec, "f1": f1,
        "latency_per_art_ms": elapsed * 1000 / total,
        "details": details
    })

# Save to logs
out_file = BASE_DIR / "logs" / "strategy_comparison_20_news.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(summary_stats, f, indent=2, ensure_ascii=False)

print(f"\nPełne wyniki zapisano do: {out_file}")
