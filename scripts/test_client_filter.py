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
        "question": "Did the article describe a financial crime committed using artificial intelligence (AI)? Answer strictly Yes or No.",
        "expected": "Yes"
    },
    {
        "name": "CASE 2: Hard Negative (Financial Crime, but NO AI)",
        "text": "A senior accountant was arrested after embezzling 12 million dollars by forging paper signatures on bank checks and altering ledger entries manually over five years.",
        "question": "Did the article describe a financial crime committed using artificial intelligence (AI)? Answer strictly Yes or No.",
        "expected": "No"
    },
    {
        "name": "CASE 3: Hard Negative (AI used, but NOT financial crime)",
        "text": "A political operative created deceptive AI deepfake video clips of an opposition candidate to spread false rumors during an election debate on social media.",
        "question": "Did the article describe a financial crime committed using artificial intelligence (AI)? Answer strictly Yes or No.",
        "expected": "No"
    },
    {
        "name": "CASE 4: Hard Negative (AI in Finance, but LEGAL / NO crime)",
        "text": "Quantitative hedge fund Renaissance Technologies reported strong quarterly returns after deploying an advanced AI neural network to optimize algorithmic stock trades on NASDAQ.",
        "question": "Did the article describe a financial crime committed using artificial intelligence (AI)? Answer strictly Yes or No.",
        "expected": "No"
    }
]

print("=" * 80)
print("  TEST STANDARDOWEGO TRYBU BOOLEAN (BEZPOŚREDNIE PYTANIE FILTRUJĄCE)")
print("=" * 80)

for c in cases:
    res = model.generate_response(c["text"], c["question"], mode="boolean")
    dec = res["generated_text"]
    conf = res["confidence"]
    p_yes = res["probabilities"]["yes"]
    p_no = res["probabilities"]["no"]
    ok = "PASS" if dec.strip().lower() == c["expected"].lower() else "FAIL"
    print(f"[{ok}] {c['name']}")
    print(f"       Decyzja: '{dec}' (Oczekiwano: '{c['expected']}')")
    print(f"       Pewność: {conf*100:.2f}% | P(Yes): {p_yes*100:.2f}% | P(No): {p_no*100:.2f}%\n")
