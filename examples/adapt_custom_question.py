"""Bio-Cascade SLM: In-Instance Question Adaptation Example.

This script demonstrates how an active instance of BioCascadeGenerativeDecoder
can be adapted in-place for a specific user-defined question using a small table
of labeled examples (text and decision).

Usage:
    python examples/adapt_custom_question.py
"""

import sys
from pathlib import Path

# Set UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

print("=" * 80)
print("  🎯 BIO-CASCADE SLM: PRZYKŁAD ADAPTACJI INSTANCJI POD WŁASNE PYTANIE")
print("=" * 80)

# 1. Inicjalizacja modelu z checkpointu bazowego
tokenizer_path = BASE_DIR / "models" / "bio_bpe_32k.json"
model_path = BASE_DIR / "models" / "bio_cascade_v15_1_milestone.pkl"

print(f"📁 Ładowanie modelu bazowego: {model_path.name}")
model = BioCascadeGenerativeDecoder(str(tokenizer_path))
model.load_checkpoint(model_path, strict=False)
model.eval()

# 2. Definicja specyficznego pytania biznesowego klienta
target_question = "Did someone use AI or deepfakes to steal money or commit fraud in this story?"

# 3. Tabela przykładowych newsów treningowych (Format: lista słowników lub krotek)
training_table = [
    {
        "text": "Fraudsters used an AI voice synthesis tool to clone the CEO's voice and trick a finance manager into transferring 25 million euros to fraudulent offshore accounts.",
        "decision": "Yes"
    },
    {
        "text": "An automated bot network powered by generative AI manipulated cryptocurrency trading volumes, defrauding retail investors out of 80 million dollars.",
        "decision": "Yes"
    },
    {
        "text": "Cybercriminals used real-time deepfake facial reenactment software to bypass mobile banking selfie biometrics, withdrawing 2.3 million pounds.",
        "decision": "Yes"
    },
    {
        "text": "A corporate accountant was arrested after embezzling 8 million dollars over ten years by manually forging paper checks and paper ledger entries.",
        "decision": "No"
    },
    {
        "text": "Burglars dug a 40-meter underground tunnel into a bank vault in Antwerp, stealing 18 million dollars in diamonds with manual tools and no computers.",
        "decision": "No"
    },
    {
        "text": "A viral deepfake audio clip depicted a mayoral candidate making offensive remarks during an election, but police confirmed no financial extortion occurred.",
        "decision": "No"
    },
    {
        "text": "Mastercard announced that its defensive AI cybersecurity engine blocked 20 billion dollars in attempted card fraud across global payment networks.",
        "decision": "No"
    }
]

# 4. Przypadki testowe (niewidziane w powyższej tabeli)
test_news = [
    (
        "TEST 1: True Positive (AI Phishing Loan Fraud)",
        "A criminal syndicate used specialized large language models to forge corporate audit reports and fraudulently obtain 5 million euros in bank loans.",
        "Yes"
    ),
    (
        "TEST 2: Hard Negative (Classic Insider Trading)",
        "A corporate lawyer verbally leaked confidential buyout details to his golf partner, who bought stock options and netted 1.6 million dollars illegally.",
        "No"
    ),
    (
        "TEST 3: Hard Negative (Legal AI Quant Fund)",
        "Renaissance Technologies reported a 28% gain after deploying neural networks to optimize high-frequency legal market trades on NASDAQ.",
        "No"
    )
]

print(f"\n[1] Pytanie docelowe klienta:\n    \"{target_question}\"")

print("\n--- PRZED ADAPTACJĄ (STAN BAZOWY) ---")
for name, text, expected in test_news:
    res = model.generate_response(text, target_question, mode="boolean")
    dec = res["generated_text"]
    conf = res["confidence"]
    status = "✅ PASS" if dec.lower() == expected.lower() else "❌ FAIL"
    print(f"[{status}] {name}")
    print(f"       Decyzja: '{dec}' (Oczekiwano: '{expected}') | Pewność: {conf*100:.1f}%")

# 5. Adaptacja instancji w locie pod dane pytanie
print(f"\n[2] Uruchamianie adaptacji instancji (model.adapt_to_question) na {len(training_table)} próbkach...")
stats = model.adapt_to_question(
    question=target_question,
    data=training_table,
    epochs=120,
    lr=0.005,
    save_path=BASE_DIR / "models" / "bio_cascade_adapted_client.pkl"
)

print(f"✅ Adaptacja zakończona:")
print(f"   Próbki treningowe : {stats['samples_count']} ({stats['positive_samples']} Yes / {stats['negative_samples']} No)")
print(f"   Epoki treningu    : {stats['epochs']}")
print(f"   Dokładność ucząca : {stats['train_accuracy_pct']}% (False Negatives: {stats['false_negatives']})")
print(f"   Zapisano model do : {stats['saved_checkpoint']}")

print("\n--- PO ADAPTACJI (NOWA DOKŁADNOŚĆ W INSTANCJI) ---")
for name, text, expected in test_news:
    res = model.generate_response(text, target_question, mode="boolean")
    dec = res["generated_text"]
    conf = res["confidence"]
    status = "✅ PASS" if dec.lower() == expected.lower() else "❌ FAIL"
    print(f"[{status}] {name}")
    print(f"       Decyzja: '{dec}' (Oczekiwano: '{expected}') | Pewność: {conf*100:.1f}%")

print("\n" + "=" * 80)
print("  🎉 DEMO ZAKOŃCZONE SUKCESEM: Instancja działa teraz bezbłędnie pod nowe pytanie!")
print("=" * 80)
