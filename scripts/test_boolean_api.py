import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder

slm = BioCascadeGenerativeDecoder("models/bio_bpe_32k.json")
slm.load_checkpoint("models/bio_cascade_v15_1_milestone.pkl", strict=False)
slm.eval()

ctx = (
    "Founded twelve years ago in Geneva, Aether Dynamics announced record quarterly revenues "
    "reaching 42.85 million euros. Meanwhile, its subsidiary Helios Energy focused on smart-grid "
    "optimization in Munich."
)

print("=" * 80)
print("TEST 1: Explicit mode='boolean' - True proposition")
print("=" * 80)
res1 = slm.generate_response(ctx, "Aether Dynamics announced record quarterly revenues.", mode="boolean")
print(json.dumps(res1, indent=2, ensure_ascii=False))

print("\n" + "=" * 80)
print("TEST 2: Explicit mode='boolean' - False proposition")
print("=" * 80)
res2 = slm.generate_response(ctx, "Helios Energy was founded in Tokyo.", mode="boolean")
print(json.dumps(res2, indent=2, ensure_ascii=False))

print("\n" + "=" * 80)
print("TEST 3: Polish question with explicit mode='tak_nie'")
print("=" * 80)
ctx_pl = "Firma Aether Dynamics osiągnęła rekordowe przychody na poziomie 42 milionów euro."
q_pl = "Czy Aether Dynamics osiągnęła rekordowe przychody?"
res3 = slm.generate_response(ctx_pl, q_pl, mode="tak_nie")
print(json.dumps(res3, indent=2, ensure_ascii=False))
