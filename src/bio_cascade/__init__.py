"""Bio-Cascade SLM compatibility forwarder to decision_flea."""

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
