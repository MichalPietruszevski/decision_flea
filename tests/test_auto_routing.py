import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

def determine_mode(question_text: str) -> str:
    q = question_text.lower().strip()
    # 1. Booleans
    if any(t in q for t in ["yes or no", "tak czy nie", "tak lub nie", "prawda czy fałsz"]):
        return "word"
    # 2. Explanatory & Clause Questions
    clause_triggers = [
        "how did", "how does", "why did", "why does", "why is", "why was", "summarize", "describe", "explain",
        "what role", "what was the concept", "what did", "what was the arche", "what core",
        "what dangerous", "what specific safety", "what limitation", "what radical", "what hypothesis",
        "what happens", "what gives", "what is the doctrine", "what is the theory", "what mechanism",
        "what trade-off", "under what condition", "how does the", "what did the transition"
    ]
    if any(q.startswith(t) or f" {t} " in f" {q} " for t in clause_triggers):
        return "paragraph"
    if any(w in q for w in ["differ from", "signify in", "function during", "doctrine of", "theory of", "vulnerability was"]):
        return "paragraph"
    # 3. Default to word for factoids (who, which company, what year, in what year, how many...)
    return "word"

model = BioCascadeGenerativeDecoder(str(BASE_DIR / "models" / "bio_bpe_32k.json"))
model.load_checkpoint(BASE_DIR / "models" / "bio_cascade_generative_live.pkl")
model.eval()

# Test a few regression questions
q_reg1 = "Who was known for his military expansions after Duke Mieszko I?"
m1 = determine_mode(q_reg1)
print(f"Regression Q1: mode={m1} -> Out: {model.generate_response('Duke Mieszko I was succeeded by Bolesław I the Brave.', q_reg1, mode=m1)['generated_text']}")

q_reg2 = "In what year did the medieval kingdom formally begin with Bolesław I?"
m2 = determine_mode(q_reg2)
print(f"Regression Q2: mode={m2} -> Out: {model.generate_response('In 1025, the medieval kingdom formally began.', q_reg2, mode=m2)['generated_text']}")

q_expl = "What primary civic and national role did visual art serve in 19th-century Poland during the partitions?"
m3 = determine_mode(q_expl)
print(f"Explanatory Q: mode={m3}")
