"""Decision-Flea SLM: Grounded Cascade-Correlation Small Language Model.

Public API:
    - DecisionFlea: Unified client and pipeline interface.
    - DecisionFleaPipeline: Alias for DecisionFlea.
    - DecisionFleaSettings: Configuration container.
    - OpenAIAssistantBridge: Bridge to OpenAI-standard models (gpt-4o, vLLM, Ollama).
    - MockOpenAIAssistant: Offline simulation assistant for testing without API keys.
"""

from src.decision_flea.pipeline import (
    DecisionFlea,
    DecisionFleaPipeline,
    BioCascade,
    BioCascadePipeline,
)
from src.decision_flea.settings import DecisionFleaSettings, BioCascadeSettings
from src.decision_flea.assistant import OpenAIAssistantBridge, MockOpenAIAssistant

__version__ = "15.1.0"
__all__ = [
    "DecisionFlea",
    "DecisionFleaPipeline",
    "DecisionFleaSettings",
    "BioCascade",
    "BioCascadePipeline",
    "BioCascadeSettings",
    "OpenAIAssistantBridge",
    "MockOpenAIAssistant",
]
