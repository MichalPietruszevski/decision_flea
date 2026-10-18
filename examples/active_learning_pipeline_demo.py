"""Bio-Cascade SLM: Autonomous Active Learning & OpenAI Assistant Pipeline Demo.

This script demonstrates:
1. Importing BioCascade as a clean library (`import bio_cascade`).
2. Configuring execution via `.settings()` with:
   - Pandas DataFrame (`df`)
   - Text and decision column mappings
   - Custom business question (e.g. AI Financial Crime)
   - OpenAI standard assistant teacher (offline mock or live gpt-4o)
3. Self-healing execution loop:
   - Bootstrap phase: Calibrates SLM with assistant until accuracy threshold is met.
   - High-speed SLM run: 15-25 ms per row on CPU.
   - Autonomous Spot-Checking: Randomly samples rows to compare against assistant.
   - Drift Recovery & Rollback: Upon any disagreement, retrains L6 in ~2s and rolls back
     preceding rows to re-verify them automatically.

Usage:
    python examples/active_learning_pipeline_demo.py
"""

import sys
from pathlib import Path
import pandas as pd

# Set UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

import bio_cascade
from bio_cascade import BioCascade, MockOpenAIAssistant, OpenAIAssistantBridge

print("=" * 80)
print("  🚀 BIO-CASCADE SLM: AUTONOMOUS ACTIVE LEARNING & ASSISTANT PIPELINE")
print("=" * 80)

# 1. Przykładowa tabela newsów (Pandas DataFrame)
news_corpus = [
    # Bootstrap / Seed items
    {"id": 1, "article": "Fraudsters used an AI voice synthesis tool to clone the CEO's voice and trick a finance manager into transferring 25 million euros."},
    {"id": 2, "article": "An automated bot network powered by generative AI manipulated cryptocurrency trading volumes, defrauding retail investors out of 80M."},
    {"id": 3, "article": "A corporate accountant was arrested after embezzling 8 million dollars over ten years by manually forging paper checks and ledger entries."},
    {"id": 4, "article": "Burglars dug a 40-meter underground tunnel into a bank vault in Antwerp, stealing 18 million dollars in diamonds with manual tools."},
    {"id": 5, "article": "Cybercriminals used real-time deepfake facial reenactment software to bypass mobile banking selfie biometrics, withdrawing 2.3M."},
    {"id": 6, "article": "Mastercard announced that its defensive AI cybersecurity engine blocked 20 billion dollars in attempted card fraud across global networks."},
    
    # Fast Run Articles
    {"id": 7, "article": "A criminal syndicate used specialized large language models to forge corporate audit reports and fraudulently obtain 5M bank loans."},
    {"id": 8, "article": "A corporate lawyer verbally leaked confidential buyout details to his golf partner, who bought stock options and netted 1.6M illegally."},
    {"id": 9, "article": "Renaissance Technologies reported a 28% gain after deploying neural networks to optimize high-frequency legal market trades on NASDAQ."},
    {"id": 10, "article": "Scammers generated realistic synthetic audio of a kidnapped child demanding 50k ransom from parents via instant wire transfer."},
    {"id": 11, "article": "A bank teller stole 15,000 dollars directly from the cash drawer during afternoon shift closing without using any software."},
    {"id": 12, "article": "Autonomous algorithmic AI agents created fake shell company invoices to steal 12 million dollars from municipal treasury accounts."}
]

df = pd.DataFrame(news_corpus)
print(f"📊 Załadowano tabelę wejściową ({len(df)} artykułów).")

# 2. Definicja pytania biznesowego
target_question = "Did someone use AI or deepfakes to steal money or commit fraud in this story?"

# 3. Wybór asystenta (OpenAI standard)
# Opcja A (Live OpenAI):
# assistant = OpenAIAssistantBridge(api_key="sk-...", model="gpt-4o-mini")
# Opcja B (Offline Mock do testów bez ponoszenia kosztów API):
assistant = MockOpenAIAssistant()

print(f"\n[1] Pytanie docelowe klienta:\n    \"{target_question}\"")
print("[2] Inicjalizacja klienta BioCascade...")

pipeline = BioCascade()

# 4. Konfiguracja parametrów przez metodę .settings(...)
pipeline.settings(
    question=target_question,
    dataframe=df,
    text_column="article",
    decision_column="slm_decision",
    confidence_column="slm_confidence",
    assistant=assistant,
    min_bootstrap_samples=4,    # Początkowa liczba próbek z asystenta
    target_accuracy=0.95,       # Docelowa dokładność kalibracji
    spot_check_rate=0.40,       # 40% wyrywkowych kontroli jakości z asystentem
    rollback_window=3,          # W razie wykrycia błędu cofnij i zweryfikuj 3 poprzednie wiersze
    epochs=120,
    recovery_epochs=120,
    verbose=True
)

# 5. Uruchomienie samouczącego się przetwarzania
print("\n[3] Uruchamianie przetwarzania tabeli...")
annotated_df, telemetry = pipeline.run()

print("\n" + "=" * 80)
print("  📋 WYNIKI PRZETWARZANIA TABELI (PANDAS DATAFRAME)")
print("=" * 80)

display_cols = ["id", "slm_decision", "slm_confidence", "spot_checked", "assistant_decision"]
print(annotated_df[display_cols].to_string(index=False))

print("\n" + "=" * 80)
print("  📈 RAPORT TELEMETRII I OSZCZĘDNOŚCI KOSZTÓW API")
print("=" * 80)
for k, v in telemetry.items():
    print(f"  - {k:<25} : {v}")

print("=" * 80)
print("  🎉 PIPELINE ZAKOŃCZONY SUKCESEM!")
print("=" * 80)
