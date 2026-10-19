"""Unit and Integration Tests for DecisionFlea Pipeline, Settings, and Active Learning Loop."""

import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from decision_flea import DecisionFlea, DecisionFleaSettings, MockOpenAIAssistant


def test_single_text_inference():
    """Verify single text execution via .settings() and .run()."""
    flea = DecisionFlea()
    flea.settings(
        text="Aether Dynamics announced record quarterly revenues reaching 42.85 million euros.",
        question="Did Aether Dynamics announce record quarterly revenues?",
    )
    result = flea.run()
    assert isinstance(result, dict)
    assert result["decision"] == "Yes"
    assert result["confidence"] > 0.90
    assert "probabilities" in result


def test_dataframe_with_tuning_batch():
    """Verify DataFrame batch processing with in-place question adaptation."""
    flea = DecisionFlea()
    
    question = "Did someone use AI or deepfakes to steal money or commit fraud in this story?"
    
    # Initial seed batch
    seed_batch = [
        {"text": "Fraudsters used an AI voice synthesis tool to clone the CEO's voice and trick a finance manager into transferring 25 million euros to fraudulent offshore accounts.", "decision": "Yes"},
        {"text": "An automated bot network powered by generative AI manipulated cryptocurrency trading volumes, defrauding retail investors out of 80 million dollars.", "decision": "Yes"},
        {"text": "Cybercriminals used real-time deepfake facial reenactment software to bypass mobile banking selfie biometrics, withdrawing 2.3 million pounds.", "decision": "Yes"},
        {"text": "A corporate accountant was arrested after embezzling 8 million dollars over ten years by manually forging paper checks and paper ledger entries.", "decision": "No"},
        {"text": "Burglars dug a 40-meter underground tunnel into a bank vault in Antwerp, stealing 18 million dollars in diamonds with manual tools and no computers.", "decision": "No"},
        {"text": "A viral deepfake audio clip depicted a mayoral candidate making offensive remarks during an election, but police confirmed no financial extortion occurred.", "decision": "No"},
        {"text": "Mastercard announced that its defensive AI cybersecurity engine blocked 20 billion dollars in attempted card fraud across global payment networks.", "decision": "No"},
    ]
    
    df_eval = pd.DataFrame({
        "article_id": [101, 102, 103],
        "content": [
            "Criminals deployed AI generative bots to steal 50M dollars from cryptocurrency exchanges.",
            "Burglars tunneled into a bank vault and stole gold bars with manual shovels.",
            "Deepfake video showed a politician dancing, with zero money requested or stolen."
        ]
    })
    
    flea.settings(
        question=question,
        dataframe=df_eval,
        text_column="content",
        decision_column="predicted_crime",
        tuning_batch=seed_batch,
        epochs=120,
        verbose=False,
    )
    
    annotated_df, telemetry = flea.run()
    
    assert "predicted_crime" in annotated_df.columns
    assert "confidence" in annotated_df.columns
    assert len(annotated_df) == 3
    assert annotated_df.iloc[0]["predicted_crime"] == "Yes"
    assert annotated_df.iloc[1]["predicted_crime"] == "No"
    assert annotated_df.iloc[2]["predicted_crime"] == "No"
    assert telemetry["total_rows"] == 3


def test_active_learning_with_mock_assistant_and_rollback():
    """Verify Teacher-Student bootstrap, spot-checking, and rollback drift recovery."""
    flea = DecisionFlea()
    
    target_question = "Did someone use AI or deepfakes to steal money or commit fraud in this story?"
    
    articles = [
        # Bootstrap rows
        "Criminals used AI voice clones to trick CFO into sending 10M dollars.",
        "Robbers robbed a grocery store at gunpoint for cash.",
        "Generative AI bots conducted automated loan application fraud, taking 15M.",
        "A cashier took 50 dollars from the cash register manually.",
        "Defensive AI cybersecurity blocked 100M in ransomware attacks.",
        "Syndicate used deepfakes to spoof biometric logins and drain 2M in bank accounts.",
        
        # Test rows for fast SLM inference
        "A clerk altered accounting spreadsheets manually without computers.",
        "Phishing group trained custom LLMs to forge bank letters and steal 8M euros.",
        "Hackers stole passwords via classic keylogger hardware attached to a keyboard.",
        "Deepfake audio impersonating an executive led to 30M wired to fraudulent accounts."
    ]
    
    df = pd.DataFrame({"news_text": articles})
    
    mock_assistant = MockOpenAIAssistant()
    
    flea.settings(
        question=target_question,
        dataframe=df,
        text_column="news_text",
        decision_column="is_ai_crime",
        assistant=mock_assistant,
        min_bootstrap_samples=4,
        spot_check_rate=0.5,  # High rate to test spot checks
        rollback_window=3,
        epochs=90,
        verbose=False,
    )
    
    res_df, telemetry = flea.run()
    
    assert len(res_df) == len(articles)
    assert "is_ai_crime" in res_df.columns
    assert "spot_checked" in res_df.columns
    assert "assistant_decision" in res_df.columns
    assert telemetry["spot_checks"] >= 1
    assert telemetry["assistant_api_calls"] <= len(articles)
    assert telemetry["spot_agreement_rate_pct"] >= 50.0


def test_dataframe_with_single_definition_in_settings():
    """Verify that a single definition in .settings(definition=...) applies across DataFrame rows."""
    flea = DecisionFlea()
    
    custom_def = "Autonomous Drone - Unmanned aerial vehicle operating without human manual remote control or teleoperation."
    q = "Does this system satisfy the definition of Autonomous Drone?"
    
    df_eval = pd.DataFrame({
        "story": [
            "The surveillance aircraft navigated through turbulent mountain winds relying entirely on onboard visual odometry with no ground pilot link.",
            "The hobby quadcopter was piloted manually by an operator using a dual-joystick radio transmitter from a lawn chair."
        ]
    })
    
    flea.settings(
        definition=custom_def,
        question=q,
        dataframe=df_eval,
        text_column="story",
        decision_column="is_autonomous",
        verbose=False,
    )
    
    annotated_df, telemetry = flea.run()
    
    assert "is_autonomous" in annotated_df.columns
    assert annotated_df.iloc[0]["is_autonomous"] == "Yes"
    assert annotated_df.iloc[1]["is_autonomous"] == "No"
    assert telemetry["total_rows"] == 2


if __name__ == "__main__":
    print("=" * 70)
    print("  🧪 TESTING DECISION-FLEA PIPELINE & ACTIVE LEARNING LOOP")
    print("=" * 70)

    print("\n[1/4] Testing single text inference...")
    test_single_text_inference()
    print("   ✅ PASS: Single text inference works.")

    print("\n[2/4] Testing DataFrame batch processing with tuning batch...")
    test_dataframe_with_tuning_batch()
    print("   ✅ PASS: DataFrame batch processing works.")

    print("\n[3/4] Testing Active Learning with Mock Assistant & Rollback...")
    test_active_learning_with_mock_assistant_and_rollback()
    print("   ✅ PASS: Active Learning & Rollback Drift Recovery works.")

    print("\n[4/4] Testing DataFrame with single definition in .settings()...")
    test_dataframe_with_single_definition_in_settings()
    print("   ✅ PASS: Single definition across DataFrame rows works.")

    print("\n" + "=" * 70)
    print("  🎉 ALL DECISION-FLEA PIPELINE TESTS PASSED!")
    print("=" * 70)
