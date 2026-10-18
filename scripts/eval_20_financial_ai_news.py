import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

# Initialize model
tokenizer_path = BASE_DIR / "models" / "bio_bpe_32k.json"
model_path = BASE_DIR / "models" / "bio_cascade_v15_2_fin_ai.pkl"

print("=" * 85)
print("  🧪 EWALUACJA: 20 SZTUCZNYCH NEWSÓW - PRZESTĘPSTWA FINANSOWE Z UŻYCIEM AI")
print("     [Naturalne pytania analityka | True Positives vs Adversarial Hard Negatives]")
print("=" * 85)

model = BioCascadeGenerativeDecoder(str(tokenizer_path))
model.load_checkpoint(model_path, strict=False)
model.eval()

# 20 diverse, realistic synthetic news articles
dataset = [
    # --- GRUPA 1: TRUE POSITIVES (Przestępstwa finansowe z użyciem AI) [8 artykułów] ---
    {
        "id": 1,
        "category": "TP (AI Voice Deepfake Wire Fraud)",
        "headline": "Multinational engineering firm loses 25 million euros after deepfake CFO call",
        "text": (
            "Detectives in Frankfurt are investigating a massive corporate heist where cybercriminals "
            "used an AI voice synthesis model to clone the voice of the chief financial officer. "
            "The synthetic audio convinced a branch comptroller to authorize five emergency wire "
            "transfers totaling 25.4 million euros to fraudulent offshore bank accounts in Vanuatu."
        ),
        "question": "Did someone use AI or deepfakes to steal money or commit fraud in this story?",
        "expected": "Yes"
    },
    {
        "id": 2,
        "category": "TP (AI Crypto Ponzi & Algorithmic Manipulation)",
        "headline": "SEC charges founders of BitQuantum with 80M dollar algorithmic trading scam",
        "text": (
            "Federal prosecutors announced indictments against two programmers who marketed BitQuantum, "
            "a fraudulent cryptocurrency scheme. The founders deployed an automated generative AI bot "
            "network to falsify trading volumes and siphon 80 million dollars in victim deposits into "
            "their private crypto wallets while fabricating bogus returns."
        ),
        "question": "Does this article talk about a financial scam carried out with artificial intelligence?",
        "expected": "Yes"
    },
    {
        "id": 3,
        "category": "TP (AI Video Clone Investment Trap)",
        "headline": "Retirees lose life savings to deepfake investment scheme on social media",
        "text": (
            "Police in Sydney issued an urgent warning after over 300 pensioners lost a collective "
            "14 million Australian dollars. Scammers used AI video cloning tools to create fake video "
            "broadcasts of banking chiefs endorsing a non-existent high-yield bond fund, funneling all "
            "sent funds to illicit shell companies."
        ),
        "question": "Was artificial intelligence used to pull off a financial theft or fraud here?",
        "expected": "Yes"
    },
    {
        "id": 4,
        "category": "TP (AI Phishing Generating Fraudulent Bank Loans)",
        "headline": "Gang dismantled after using LLMs to forge identities and steal 5M in bank loans",
        "text": (
            "Spanish police arrested six individuals who utilized specialized large language models to "
            "synthesize thousands of authentic-looking financial audit statements and forged paystubs. "
            "The gang fraudulently obtained over 5 million euros in unsecured bank loans before disappearing."
        ),
        "question": "Did fraudsters use AI tools to illegally steal funds or commit loan fraud?",
        "expected": "Yes"
    },
    {
        "id": 5,
        "category": "TP (Automated AI Market Spoofing)",
        "headline": "Rogue trader fined 12M for training neural network to manipulate commodities",
        "text": (
            "The CFTC penalized a former energy trader who programmed a deep learning algorithm to "
            "submit and cancel thousands of fictitious oil futures orders in milliseconds. The predatory "
            "AI spoofing bot manipulated prices, illegally netting the rogue trader 3.8 million dollars in profits."
        ),
        "question": "Does this story describe an illegal financial market crime executed by an AI algorithm?",
        "expected": "Yes"
    },
    {
        "id": 6,
        "category": "TP (Deepfake Facial Biometric Bypass on Banking App)",
        "headline": "Cyber-gang bypasses mobile banking facial recognition to drain accounts",
        "text": (
            "A security audit revealed that an organized crime group used real-time generative AI face "
            "reenactment software to bypass selfie biometric verification on digital banking apps, "
            "illicitly withdrawing over 2.3 million pounds from compromised customer accounts."
        ),
        "question": "Did criminals use generative AI software to steal money from bank accounts?",
        "expected": "Yes"
    },
    {
        "id": 7,
        "category": "TP (AI Botnet Ransomware on Payment Gateway)",
        "headline": "Fintech payment processor extorted for 10M after self-learning malware strike",
        "text": (
            "A European fintech payment gateway suffered an attack by autonomous AI-driven ransomware "
            "that adapted its evasion code in real-time. The extortionists encrypted payment pipelines "
            "and blackmailed the company into paying a 10 million dollar ransom in Monero."
        ),
        "question": "Was an adaptive AI tool utilized to extort and steal money in this incident?",
        "expected": "Yes"
    },
    {
        "id": 8,
        "category": "TP (AI Automated Credit Card Testing & Laundering)",
        "headline": "Online marketplace flooded by AI agents laundering stolen credit card balances",
        "text": (
            "Financial investigators uncovered an underground operation where autonomous AI agents "
            "were programmed to micro-test thousands of stolen credit card numbers against digital gift "
            "cards, successfully laundering 4.5 million dollars through e-commerce accounts."
        ),
        "question": "Did criminals program AI agents to execute a financial theft or money laundering scheme?",
        "expected": "Yes"
    },

    # --- GRUPA 2: HARD NEGATIVE: Przestępstwa finansowe, ale BEZ AI [5 artykułów] ---
    {
        "id": 9,
        "category": "HN (Traditional Paper Embezzlement)",
        "headline": "Senior accountant jailed after stealing 8M through forged paper checks",
        "text": (
            "A veteran payroll manager was sentenced to nine years in prison after embezzling 8.2 million "
            "dollars from a logistics company over a decade. The accountant manually forged executive "
            "signatures on paper bank checks and falsified handwritten physical ledger books."
        ),
        "question": "Did someone use AI or deepfakes to steal money or commit fraud in this story?",
        "expected": "No"
    },
    {
        "id": 10,
        "category": "HN (Classic Insider Trading via Human Leaks)",
        "headline": "Corporate lawyer arrested for tipping golf partner on pending pharma merger",
        "text": (
            "Federal agents arrested a merger attorney who verbally leaked confidential buyout details "
            "to his childhood friend during a private golf game. The friend purchased call options and "
            "netted 1.6 million dollars in illegal insider trading profits before the acquisition was public."
        ),
        "question": "Does this article talk about a financial scam carried out with artificial intelligence?",
        "expected": "No"
    },
    {
        "id": 11,
        "category": "HN (Physical Bank Robbery / Vault Drilling)",
        "headline": "Thieves tunnel into underground safe deposit vault and escape with diamonds",
        "text": (
            "In a daring weekend heist in Antwerp, burglars excavated a 40-meter underground tunnel from "
            "a nearby sewer to access a private deposit vault. The thieves disabled alarm cables by hand "
            "and escaped with 18 million dollars worth of uncut diamonds and cash."
        ),
        "question": "Was artificial intelligence used to pull off a financial theft or fraud here?",
        "expected": "No"
    },
    {
        "id": 12,
        "category": "HN (Traditional Real Estate Ponzi Scheme)",
        "headline": "Developer sentenced for running 50M dollar suburban property pyramid",
        "text": (
            "Florida real estate promoter Arthur Vance received twelve years for orchestrating a classic "
            "Ponzi scheme. Vance promised investors 20% annual returns on condominium developments, using "
            "new investor capital to pay off older investors while squandering the remainder on private yachts."
        ),
        "question": "Did fraudsters use AI tools to illegally steal funds or commit loan fraud?",
        "expected": "No"
    },
    {
        "id": 13,
        "category": "HN (Tax Evasion via Swiss Offshore Shells)",
        "headline": "Oil executive indicted for concealing 35M in undisclosed Liechtenstein trusts",
        "text": (
            "Justice Department officials filed criminal tax evasion charges against an oil magnate who "
            "hid 35 million dollars in offshore trusts. The evasion was conducted through paper bearer shares "
            "and shell corporations established by Zurich attorneys without any computer automation."
        ),
        "question": "Does this story describe an illegal financial market crime executed by an AI algorithm?",
        "expected": "No"
    },

    # --- GRUPA 3: HARD NEGATIVE: Użycie AI, ale BEZ przestępstwa finansowego [4 artykuły] ---
    {
        "id": 14,
        "category": "HN (AI Deepfake in Political Campaign / No Money)",
        "headline": "Controversy erupts as deepfake audio of mayoral candidate surfaces online",
        "text": (
            "City council elections were thrown into turmoil after a viral audio clip generated by an AI "
            "model falsely depicted the incumbent mayor insulting civil service employees. The police opened "
            "a defamation inquiry into the political prank, but confirmed no money or assets were targeted."
        ),
        "question": "Did someone use AI or deepfakes to steal money or commit fraud in this story?",
        "expected": "No"
    },
    {
        "id": 15,
        "category": "HN (AI in Medical Imaging Research)",
        "headline": "Hospital tests deep learning network to detect pancreatic cancer early",
        "text": (
            "Researchers at John Hopkins Hospital reported breakthrough results from an AI diagnostic "
            "model that identifies early-stage tumors on CT scans with 96% accuracy. The clinical trial "
            "involved 5,000 anonymized patient scans and was hailed by oncologists."
        ),
        "question": "Does this article talk about a financial scam carried out with artificial intelligence?",
        "expected": "No"
    },
    {
        "id": 16,
        "category": "HN (Autonomous Drone Traffic Incident)",
        "headline": "Delivery drone crashes into residential power line due to navigation glitch",
        "text": (
            "An autonomous package delivery drone operated by SkyRoute crashed into an overhead electrical "
            "line during heavy fog, causing a temporary blackout for 200 homes. Company engineers stated "
            "the on-board computer vision algorithm misclassified the wire as a tree branch."
        ),
        "question": "Was artificial intelligence used to pull off a financial theft or fraud here?",
        "expected": "No"
    },
    {
        "id": 17,
        "category": "HN (AI Video Game Generation)",
        "headline": "Indie studio launches fantasy RPG featuring AI-generated narrative quests",
        "text": (
            "Game development studio PixelCraft released a role-playing game where non-player characters "
            "use generative AI to create dynamic dialogue trees. The game received critical acclaim on Steam, "
            "selling 100,000 copies in its opening weekend."
        ),
        "question": "Did criminals use generative AI software to steal money from bank accounts?",
        "expected": "No"
    },

    # --- GRUPA 4: HARD NEGATIVE: AI w finansach, ale całkowicie LEGALNE [3 artykuły] ---
    {
        "id": 18,
        "category": "HN (Legal AI Quantitative Trading)",
        "headline": "Renaissance fund posts 28% gain after deploying transformer model on bonds",
        "text": (
            "Quantitative investment firm Renaissance Technologies reported market-beating annual returns "
            "after integrating an advanced AI neural network to optimize sovereign bond spreads. All trades "
            "were fully audited and certified in compliance with SEC regulatory standards."
        ),
        "question": "Does this story describe an illegal financial market crime executed by an AI algorithm?",
        "expected": "No"
    },
    {
        "id": 19,
        "category": "HN (Legal AI Fraud Detection by Banks)",
        "headline": "Mastercard reports AI defense system blocked 20 billion in attempted fraud",
        "text": (
            "Payments network Mastercard announced that its proprietary AI cybersecurity engine prevented "
            "over 20 billion dollars in suspected card fraud over the past year by analyzing transaction "
            "patterns across 125 billion annual payments in real time."
        ),
        "question": "Did someone use AI or deepfakes to steal money or commit fraud in this story?",
        "expected": "No"
    },
    {
        "id": 20,
        "category": "HN (Legal Automated Credit Underwriting)",
        "headline": "Credit union approves 5,000 small business loans using AI risk scoring",
        "text": (
            "Midwest Federal Credit Union expanded its lending program by adopting an automated machine "
            "learning model to evaluate borrower creditworthiness. The legitimate banking initiative "
            "shortened loan approval times from three weeks to forty seconds without increasing default rates."
        ),
        "question": "Was an adaptive AI tool utilized to extort and steal money in this incident?",
        "expected": "No"
    }
]

# Run evaluation
results = []
tp = fp = tn = fn = 0
latencies = []

print(f"\nUruchamianie ewaluacji na {len(dataset)} artykułach...\n")

for item in dataset:
    t0 = time.perf_counter()
    res = model.generate_response(item["text"], item["question"], mode="boolean")
    lat_ms = (time.perf_counter() - t0) * 1000
    latencies.append(lat_ms)
    
    dec = res["generated_text"].strip()
    exp = item["expected"]
    is_correct = (dec.lower() == exp.lower())
    conf = res["confidence"]
    p_yes = res["probabilities"]["yes"]
    p_no = res["probabilities"]["no"]
    
    if exp == "Yes":
        if is_correct:
            tp += 1
        else:
            fn += 1
    else:
        if is_correct:
            tn += 1
        else:
            fp += 1
            
    mark = "✅ PASS" if is_correct else "❌ FAIL"
    print(f"[{mark}] #{item['id']:02d} | Exp: {exp:>3} | Model: {dec:>3} (Conf: {conf*100:5.1f}%) | {item['category']}")
    print(f"       Q: {item['question']}")
    print(f"       Logity: {res['verif_logits']} | Latencja: {lat_ms:.1f} ms\n")
    
    results.append({
        "id": item["id"],
        "category": item["category"],
        "headline": item["headline"],
        "question": item["question"],
        "expected": exp,
        "predicted": dec,
        "is_correct": is_correct,
        "confidence": conf,
        "p_yes": p_yes,
        "p_no": p_no,
        "latency_ms": lat_ms
    })

# Compute metrics
total = len(dataset)
acc = (tp + tn) / total * 100.0
rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
spec = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
avg_lat = sum(latencies) / len(latencies)

print("=" * 85)
print("  📊 PODSUMOWANIE METRYK (BENCHMARK 20 NEWSÓW FINANSOWYCH)")
print("=" * 85)
print(f"  Macierz Pomyłek  : TP={tp} | FP={fp} | TN={tn} | FN={fn}")
print(f"  Recall (Czułość) : {rec:.2f}%  (Cel: FN=0)")
print(f"  Dokładność (Acc) : {acc:.2f}%  ({tp+tn}/{total})")
print(f"  Precyzja (Prec)  : {prec:.2f}%")
print(f"  Swoistość (Spec) : {spec:.2f}%")
print(f"  F1-Score         : {f1:.2f}%")
print(f"  F2-Score         : {f2:.2f}%")
print(f"  Średnia Latencja : {avg_lat:.1f} ms / news")
print("=" * 85)

# Save results
out_path = BASE_DIR / "logs" / "eval_20_financial_ai_news.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "metrics": {
            "total": total,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": acc, "recall": rec, "precision": prec,
            "specificity": spec, "f1": f1, "f2": f2,
            "avg_latency_ms": avg_lat
        },
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"Wyniki zapisano do: {out_path}")
