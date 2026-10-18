import sys
sys.path.insert(0, ".")
from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

model = BioCascadeGenerativeDecoder("models/bio_bpe_32k.json")
model.load_checkpoint("models/bio_cascade_v15_1_milestone.pkl", strict=False)
model.eval()

cases = [
    {
        "name": "CASE 1: True Positive (AI Deepfake Wire Fraud)",
        "text": "Fraudsters used an AI-generated deepfake voice clone of a company executive to trick an employee into transferring 25 million dollars from a bank account to fraudulent offshore entities.",
        "expected": True
    },
    {
        "name": "CASE 2: Hard Negative (Financial Crime, but NO AI)",
        "text": "A senior accountant was arrested after embezzling 12 million dollars by forging paper signatures on bank checks and altering ledger entries manually over five years.",
        "expected": False
    },
    {
        "name": "CASE 3: Hard Negative (AI used, but NOT financial crime)",
        "text": "A political operative created deceptive AI deepfake video clips of an opposition candidate to spread false rumors during an election debate on social media.",
        "expected": False
    },
    {
        "name": "CASE 4: Hard Negative (AI in Finance, but LEGAL / NO crime)",
        "text": "Quantitative hedge fund Renaissance Technologies reported strong quarterly returns after deploying an advanced AI neural network to optimize algorithmic stock trades on NASDAQ.",
        "expected": False
    }
]

print("=" * 80)
print("  EWALUACJA: MODUŁOWE FILTROWANIE KRYTERIALNE (FINANCE + AI CRIME)")
print("=" * 80)

for c in cases:
    # Krok 1: Weryfikacja wątku finansowego przestępstwa / wyłudzenia
    q_fin = "Did the article describe financial fraud, embezzlement, or theft of money? Answer strictly Yes or No."
    # Krok 2: Weryfikacja użycia AI / deepfake
    q_ai = "Did the article mention artificial intelligence, deepfakes, or machine learning? Answer strictly Yes or No."
    
    r_fin = model.generate_response(c["text"], q_fin, mode="boolean")
    r_ai = model.generate_response(c["text"], q_ai, mode="boolean")
    
    is_fin = "yes" in r_fin["generated_text"].lower()
    is_ai = "yes" in r_ai["generated_text"].lower()
    is_match = is_fin and is_ai
    combined_conf = (r_fin["confidence"] + r_ai["confidence"]) / 2.0
    
    status = "PASS" if is_match == c["expected"] else "FAIL"
    print(f"[{status}] {c['name']}")
    print(f"       1. Przestępstwo finansowe: {r_fin['generated_text']} (P={r_fin['confidence']*100:.1f}%)")
    print(f"       2. Użycie technologii AI: {r_ai['generated_text']} (P={r_ai['confidence']*100:.1f}%)")
    print(f"       -> Decyzja Końcowa: {'ZAKWALIFIKOWANY' if is_match else 'ODRZUCONY'} (Śr. pewność: {combined_conf*100:.1f}%)\n")
