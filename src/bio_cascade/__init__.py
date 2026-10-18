"""Bio-Cascade SLM: Grounded Cascade-Correlation Small Language Model.

Public API:
    - BioCascade: Unified client and pipeline interface.
    - BioCascadePipeline: Alias for BioCascade.
    - BioCascadeSettings: Configuration container.
    - OpenAIAssistantBridge: Bridge to OpenAI-standard models (gpt-4o, vLLM, Ollama).
    - MockOpenAIAssistant: Offline simulation assistant for testing without API keys.
"""

from src.bio_cascade.pipeline import BioCascade, BioCascadePipeline
from src.bio_cascade.settings import BioCascadeSettings
from src.bio_cascade.assistant import OpenAIAssistantBridge, MockOpenAIAssistant

__version__ = "15.1.0"
__all__ = [
    "BioCascade",
    "BioCascadePipeline",
    "BioCascadeSettings",
    "OpenAIAssistantBridge",
    "MockOpenAIAssistant",
]
