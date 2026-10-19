"""Decision-Flea SLM: Unified Library Pipeline and Active Learning Engine."""

import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.cascor.bio_cascade_generative import BioCascadeGenerativeDecoder
from src.decision_flea.assistant import MockOpenAIAssistant, OpenAIAssistantBridge
from src.decision_flea.settings import DecisionFleaSettings


class DecisionFlea:
    """High-level library client for Decision-Flea SLM.

    Provides a unified `.settings(...)` configuration interface for:
      - Single-text zero-hallucination verification.
      - Pandas DataFrame batch processing.
      - In-place few-shot question adaptation.
      - OpenAI-compatible teacher-student bootstrap.
      - Autonomous spot-checking and drift recovery (Active Learning).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        tokenizer_path: Optional[Union[str, Path]] = None,
        settings: Optional[DecisionFleaSettings] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.default_tokenizer = base_dir / "models" / "bio_bpe_32k.json"
        self.default_model = base_dir / "models" / "bio_cascade_v15_1_milestone.pkl"

        tok_path = tokenizer_path or self.default_tokenizer
        mod_path = model_path or self.default_model

        self.cfg = settings or DecisionFleaSettings(
            model_path=mod_path,
            tokenizer_path=tok_path,
        )

        # Initialize internal decoder
        self.model = BioCascadeGenerativeDecoder(str(tok_path))
        self.model.load_checkpoint(mod_path, strict=False)
        self.model.eval()

        # Telemetry state
        self.last_telemetry: Dict[str, Any] = {}

    def settings(self, **kwargs) -> "DecisionFlea":
        """Configure pipeline parameters.

        Args:
            text (str): Single context text to verify.
            question (str): Target question to evaluate.
            dataframe (pd.DataFrame): Pandas table of texts to process.
            text_column (str): Name of column containing context text.
            decision_column (str): Name of column to store output decisions.
            confidence_column (str): Name of column to store confidence.
            tuning_batch (list/df): Initial training samples for question adaptation.
            assistant (OpenAI client, dict, or bridge): Teacher LLM for active learning.
            target_accuracy (float): Calibration threshold before starting table run (default: 0.95).
            spot_check_rate (float): Fraction of rows to randomly verify against assistant (default: 0.05).
            rollback_window (int): Number of preceding rows to re-verify upon detecting drift (default: 5).
            epochs (int): Epochs for adaptation gradient descent (default: 120).
            lr (float): Learning rate for adaptation (default: 0.005).
            save_checkpoint (str/Path): Optional path to save adapted model weights.

        Returns:
            self: Enables fluent method chaining.
        """
        self.cfg.update(**kwargs)
        return self

    def _resolve_assistant(self, assistant_arg: Any) -> Any:
        """Resolve assistant input into a callable bridge."""
        if assistant_arg is None:
            return None
        if isinstance(assistant_arg, (OpenAIAssistantBridge, MockOpenAIAssistant)):
            return assistant_arg
        if isinstance(assistant_arg, dict):
            return OpenAIAssistantBridge(
                client=assistant_arg.get("client"),
                model=assistant_arg.get("model", "gpt-4o-mini"),
                temperature=assistant_arg.get("temperature", 0.0),
                api_key=assistant_arg.get("api_key"),
                base_url=assistant_arg.get("base_url"),
                system_prompt=assistant_arg.get("system_prompt"),
            )
        return OpenAIAssistantBridge(client=assistant_arg)

    def run(self, **override_kwargs) -> Any:
        """Execute the configured pipeline.

        Returns:
            - If single text: Dict[str, Any] with decision, confidence, and verifier logits.
            - If dataframe: Tuple[pd.DataFrame, Dict[str, Any]] with enriched DataFrame and telemetry.
        """
        if override_kwargs:
            self.cfg.update(**override_kwargs)

        # 1. Single-text mode
        if self.cfg.text is not None and self.cfg.dataframe is None:
            return self._run_single_text(self.cfg.text, self.cfg.question)

        # 2. DataFrame batch processing mode
        if self.cfg.dataframe is not None:
            return self._run_dataframe_pipeline()

        raise ValueError("No input data provided. Specify either 'text=...' or 'dataframe=...' in .settings()")

    def _run_single_text(self, text: str, question: str) -> Dict[str, Any]:
        """Perform zero-hallucination verification on a single text."""
        def_text = self.cfg.definition or ""
        res = self.model.generate_response(text, question, mode="boolean", definition_text=def_text)
        return {
            "question": question,
            "definition": def_text,
            "text": text,
            "decision": res["generated_text"],
            "confidence": res["confidence"],
            "probabilities": res["probabilities"],
            "verif_logits": res.get("verif_logits"),
            "definition_entailment": res.get("definition_entailment"),
        }

    def _run_dataframe_pipeline(self) -> Tuple[Any, Dict[str, Any]]:
        """Process a pandas DataFrame with active learning, spot-checking, and drift recovery."""
        df = self.cfg.dataframe.copy()
        text_col = self.cfg.text_column
        dec_col = self.cfg.decision_column
        conf_col = self.cfg.confidence_column
        definition = self.cfg.definition or ""
        question = self.cfg.question

        if text_col not in df.columns:
            raise KeyError(f"Column '{text_col}' not found in provided DataFrame.")

        assistant = self._resolve_assistant(self.cfg.assistant)
        start_time = time.time()

        training_pool: List[Dict[str, str]] = []
        if self.cfg.tuning_batch is not None:
            training_pool.extend(self._normalize_data(self.cfg.tuning_batch))

        api_calls = 0
        slm_inferences = 0
        spot_checks = 0
        spot_agreements = 0
        drift_recoveries = 0

        # ----------------------------------------------------------------------
        # PHASE 1: BOOTSTRAP ADAPTATION (If Assistant is configured)
        # ----------------------------------------------------------------------
        if assistant is not None:
            bootstrap_size = min(len(df), self.cfg.min_bootstrap_samples)
            if self.cfg.verbose:
                print(f"[DecisionFlea] [BOOTSTRAP] Starting Active Learning on {bootstrap_size} seed samples...")

            # Query assistant for seed labels
            for idx in range(bootstrap_size):
                row_text = str(df.iloc[idx][text_col])
                ast_q = f"Definition: {definition}\nQuestion: {question}" if definition else question
                ast_res = assistant.classify(row_text, ast_q)
                api_calls += 1
                training_pool.append({"text": row_text, "decision": ast_res["decision"]})

            # In-place adaptation on the seed pool
            self.model.adapt_to_question(
                question=question,
                data=training_pool,
                epochs=self.cfg.epochs,
                lr=self.cfg.lr,
                yes_weight=self.cfg.yes_weight,
            )

            # Check bootstrap accuracy against seed pool
            seed_correct = 0
            for item in training_pool:
                pred = self.model.generate_response(item["text"], question, mode="boolean", definition_text=definition)["generated_text"]
                if pred.lower() == item["decision"].lower():
                    seed_correct += 1
            running_acc = seed_correct / len(training_pool)

            if self.cfg.verbose:
                print(f"[DecisionFlea] [CALIBRATED] Bootstrap calibrated: {running_acc * 100:.1f}% accuracy on seed pool.")

        elif training_pool:
            if self.cfg.verbose:
                print(f"[DecisionFlea] [ADAPT] Adapting model on provided tuning_batch ({len(training_pool)} samples)...")
            self.model.adapt_to_question(
                question=question,
                data=training_pool,
                epochs=self.cfg.epochs,
                lr=self.cfg.lr,
                yes_weight=self.cfg.yes_weight,
            )

        # ----------------------------------------------------------------------
        # PHASE 2: FAST SLM INFERENCE WITH SPOT-CHECKING & ROLLBACK
        # ----------------------------------------------------------------------
        decisions: List[str] = [""] * len(df)
        confidences: List[float] = [0.0] * len(df)
        spot_checked: List[bool] = [False] * len(df)
        assistant_decisions: List[Optional[str]] = [None] * len(df)

        idx = 0
        while idx < len(df):
            row_text = str(df.iloc[idx][text_col])

            # 1. Ultra-fast SLM CPU inference
            res = self.model.generate_response(row_text, question, mode="boolean", definition_text=definition)
            slm_inferences += 1
            slm_dec = res["generated_text"]
            slm_conf = res["confidence"]

            decisions[idx] = slm_dec
            confidences[idx] = slm_conf

            # 2. Decide if this row gets spot-checked by the assistant
            is_check = False
            if assistant is not None:
                if idx >= self.cfg.min_bootstrap_samples and random.random() < self.cfg.spot_check_rate:
                    is_check = True

            if is_check and assistant is not None:
                spot_checks += 1
                spot_checked[idx] = True
                ast_q = f"Definition: {definition}\nQuestion: {question}" if definition else question
                ast_res = assistant.classify(row_text, ast_q)
                api_calls += 1
                ast_dec = ast_res["decision"]
                assistant_decisions[idx] = ast_dec

                if slm_dec.lower() == ast_dec.lower():
                    spot_agreements += 1
                else:
                    # ----------------------------------------------------------
                    # DRIFT DETECTED: RETRAIN & ROLLBACK
                    # ----------------------------------------------------------
                    drift_recoveries += 1
                    if self.cfg.verbose:
                        print(
                            f"[DecisionFlea] [WARN: DRIFT] Drift detected at row {idx}! "
                            f"(SLM: '{slm_dec}' vs Assistant: '{ast_dec}'). "
                            f"Initiating Drift Recovery & Rollback..."
                        )

                    # Add error sample to training pool
                    training_pool.append({"text": row_text, "decision": ast_dec})

                    # Perform quick in-place retraining on updated pool
                    self.model.adapt_to_question(
                        question=question,
                        data=training_pool,
                        epochs=self.cfg.recovery_epochs,
                        lr=self.cfg.lr,
                        yes_weight=self.cfg.yes_weight,
                    )

                    # Rollback window: re-evaluate preceding K rows with updated weights
                    k_rollback = self.cfg.rollback_window
                    rollback_start = max(0, idx - k_rollback)
                    if self.cfg.verbose:
                        print(f"[DecisionFlea] [ROLLBACK] Re-verifying rows {rollback_start} to {idx}...")

                    for r_idx in range(rollback_start, idx + 1):
                        r_text = str(df.iloc[r_idx][text_col])
                        r_res = self.model.generate_response(r_text, question, mode="boolean", definition_text=definition)
                        decisions[r_idx] = r_res["generated_text"]
                        confidences[r_idx] = r_res["confidence"]
                        slm_inferences += 1

            idx += 1

        elapsed = time.time() - start_time
        df[dec_col] = decisions
        df[conf_col] = confidences
        df["spot_checked"] = spot_checked
        if assistant is not None:
            df["assistant_decision"] = assistant_decisions

        if self.cfg.save_checkpoint:
            self.model.save_checkpoint(self.cfg.save_checkpoint)

        agreement_rate = (spot_agreements / spot_checks * 100.0) if spot_checks > 0 else 100.0
        api_savings = (1.0 - (api_calls / max(1, len(df)))) * 100.0

        telemetry = {
            "total_rows": len(df),
            "slm_inferences": slm_inferences,
            "assistant_api_calls": api_calls,
            "api_savings_pct": round(max(0.0, api_savings), 2),
            "spot_checks": spot_checks,
            "spot_agreements": spot_agreements,
            "spot_agreement_rate_pct": round(agreement_rate, 2),
            "drift_recovery_events": drift_recoveries,
            "total_training_samples": len(training_pool),
            "elapsed_seconds": round(elapsed, 3),
            "avg_ms_per_row": round((elapsed / max(1, len(df))) * 1000.0, 2),
        }
        self.last_telemetry = telemetry

        if self.cfg.verbose:
            print(
                f"[DecisionFlea] [DONE] Processing complete: {len(df)} rows in {elapsed:.2f}s "
                f"({telemetry['avg_ms_per_row']} ms/row). "
                f"Assistant API savings: {telemetry['api_savings_pct']}%. "
                f"Agreement: {telemetry['spot_agreement_rate_pct']}%."
            )

        return df, telemetry

    def _normalize_data(self, data: Any) -> List[Dict[str, str]]:
        """Normalize various input formats into a list of dicts."""
        normalized: List[Dict[str, str]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    txt = str(item.get("text") or item.get("content") or "")
                    dec = str(item.get("decision") or item.get("label") or "No")
                    normalized.append({"text": txt, "decision": dec})
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    normalized.append({"text": str(item[0]), "decision": str(item[1])})
        elif hasattr(data, "columns"):
            cols = list(data.columns)
            t_col = self.cfg.text_column if self.cfg.text_column in cols else cols[0]
            d_col = self.cfg.decision_column if self.cfg.decision_column in cols else (cols[1] if len(cols) > 1 else cols[0])
            for _, row in data.iterrows():
                normalized.append({"text": str(row[t_col]), "decision": str(row[d_col])})
        return normalized


# Friendly aliases
DecisionFleaPipeline = DecisionFlea
BioCascade = DecisionFlea
BioCascadePipeline = DecisionFlea
