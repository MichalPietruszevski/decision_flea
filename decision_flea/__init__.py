"""Decision-Flea SLM root module forwarder."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.decision_flea import (
    DecisionFlea,
    DecisionFleaPipeline,
    DecisionFleaSettings,
    BioCascade,
    BioCascadePipeline,
    BioCascadeSettings,
    OpenAIAssistantBridge,
    MockOpenAIAssistant,
    __version__,
)

__all__ = [
    "DecisionFlea",
    "DecisionFleaPipeline",
    "DecisionFleaSettings",
    "BioCascade",
    "BioCascadePipeline",
    "BioCascadeSettings",
    "OpenAIAssistantBridge",
    "MockOpenAIAssistant",
    "__version__",
]
