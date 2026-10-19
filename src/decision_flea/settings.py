"""Decision-Flea SLM: Settings and Configuration Dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class DecisionFleaSettings:
    """Configuration container for Decision-Flea execution pipeline.

    Supports:
        - Single text inference
        - Pandas DataFrame batch processing
        - Question adaptation with initial batch
        - OpenAI standard assistant teacher integration
        - Active learning (bootstrap & spot-check drift recovery)
    """

    # 1. Target Query / Question & Optional Definition
    question: str = ""
    definition: Optional[str] = None

    # 2. Input Data
    text: Optional[str] = None
    dataframe: Any = None
    text_column: str = "text"
    decision_column: str = "decision"
    confidence_column: str = "confidence"

    # 3. Fine-Tuning Batch / Seed Data
    tuning_batch: Any = None

    # 4. OpenAI Assistant Configuration
    assistant: Any = None
    target_accuracy: float = 0.95
    min_bootstrap_samples: int = 6
    max_bootstrap_samples: int = 30
    bootstrap_batch_size: int = 4

    # 5. Continuous Active Learning & Drift Recovery
    spot_check_rate: float = 0.05
    rollback_window: int = 5
    recovery_epochs: int = 120

    # 6. Adaptation Hyperparameters
    epochs: int = 120
    lr: float = 0.005
    weight_decay: float = 1e-5
    yes_weight: float = 1.5

    # 7. Model Checkpoints & Execution
    model_path: Optional[str | Path] = None
    tokenizer_path: Optional[str | Path] = None
    save_checkpoint: Optional[str | Path] = None
    verbose: bool = True

    def update(self, **kwargs) -> "DecisionFleaSettings":
        """Update settings with arbitrary key-value pairs."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"DecisionFleaSettings has no configuration attribute '{key}'")
        return self


# Backwards compatibility alias
BioCascadeSettings = DecisionFleaSettings
