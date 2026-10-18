"""Bio-Cascade SLM root module forwarder."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.bio_cascade import (
    BioCascade,
    BioCascadePipeline,
    BioCascadeSettings,
    OpenAIAssistantBridge,
    MockOpenAIAssistant,
    __version__,
)

__all__ = [
    "BioCascade",
    "BioCascadePipeline",
    "BioCascadeSettings",
    "OpenAIAssistantBridge",
    "MockOpenAIAssistant",
    "__version__",
]
