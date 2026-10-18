"""Bio-Cascade SLM: OpenAI Standard Assistant Bridge & Offline Mock.

Provides an interface to query any OpenAI-compatible LLM (e.g. gpt-4o, Azure OpenAI,
vLLM, Ollama, LocalAI) as an oracle / teacher assistant, with built-in prompt formatting,
binary decision extraction, and MD5 query caching.
"""

import hashlib
import json
import re
from typing import Any, Callable, Dict, Optional


class OpenAIAssistantBridge:
    """Wrapper for OpenAI-standard chat completion clients."""

    def __init__(
        self,
        client: Any = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.cache: Dict[str, Dict[str, Any]] = {}

        if client is not None:
            self.client = client
        else:
            try:
                import openai
                kwargs = {}
                if api_key:
                    kwargs["api_key"] = api_key
                if base_url:
                    kwargs["base_url"] = base_url
                self.client = openai.OpenAI(**kwargs)
            except Exception as e:
                self.client = None
                self._init_error = str(e)

        self.system_prompt = system_prompt or (
            "You are a strict binary verification assistant. "
            "Given an article context and a target question, determine whether the question is strictly answered with YES or NO. "
            "You must respond with valid JSON in this exact format: "
            '{"decision": "Yes" | "No", "confidence": float between 0.0 and 1.0, "reasoning": "brief explanation"}'
        )

    def _cache_key(self, text: str, question: str) -> str:
        content = f"{question}:::{text}".encode("utf-8")
        return hashlib.md5(content).hexdigest()

    def classify(self, text: str, question: str) -> Dict[str, Any]:
        """Classify a text with respect to the question using the assistant."""
        key = self._cache_key(text, question)
        if key in self.cache:
            return self.cache[key]

        if self.client is None:
            raise RuntimeError(
                "OpenAI client is not initialized. Please pass a valid openai client, "
                "or install openai ('pip install openai') and provide an api_key, "
                "or use MockOpenAIAssistant for offline simulation."
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Context Article:\n\"\"\"{text}\"\"\"\n\nTarget Question:\n{question}\n\nRespond with JSON:",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"} if hasattr(self.client, "chat") else None,
        )

        raw_content = response.choices[0].message.content.strip()
        decision, confidence, reasoning = self._parse_response(raw_content)

        result = {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "raw": raw_content,
            "cached": False,
        }
        self.cache[key] = {**result, "cached": True}
        return result

    def _parse_response(self, raw_content: str) -> tuple[str, float, str]:
        """Extract decision, confidence, and reasoning from assistant output."""
        try:
            data = json.loads(raw_content)
            dec = str(data.get("decision", "No")).strip().capitalize()
            decision = "Yes" if dec.startswith("Y") else "No"
            conf = float(data.get("confidence", 0.95))
            reason = str(data.get("reasoning", ""))
            return decision, conf, reason
        except Exception:
            # Fallback regex extraction
            if re.search(r'\b(yes|true)\b', raw_content, re.IGNORECASE):
                decision = "Yes"
            else:
                decision = "No"
            return decision, 0.90, raw_content


class MockOpenAIAssistant:
    """Offline mock assistant implementing the OpenAI standard interface.

    Allows running active learning, bootstrap distillation, and spot-check tests
    without requiring internet connectivity or OpenAI API credits.
    """

    def __init__(
        self,
        oracle_fn: Optional[Callable[[str, str], tuple[str, float]]] = None,
        ground_truth_map: Optional[Dict[str, str]] = None,
    ):
        self.oracle_fn = oracle_fn
        self.ground_truth_map = ground_truth_map or {}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.call_count = 0

    def _cache_key(self, text: str, question: str) -> str:
        content = f"{question}:::{text}".encode("utf-8")
        return hashlib.md5(content).hexdigest()

    def classify(self, text: str, question: str) -> Dict[str, Any]:
        self.call_count += 1
        key = self._cache_key(text, question)
        if key in self.cache:
            return self.cache[key]

        # 1. Lookup in explicit dictionary if provided
        if text in self.ground_truth_map:
            decision = self.ground_truth_map[text]
            confidence = 1.0
            reason = "Exact match in ground truth dictionary"
        # 2. Custom oracle callback function
        elif self.oracle_fn is not None:
            decision, confidence = self.oracle_fn(text, question)
            reason = "Oracle callback decision"
        # 3. Default semantic heuristic for financial AI crime / deepfakes
        else:
            lower = text.lower()
            q_lower = question.lower()
            
            # Check for financial AI crime
            has_ai = any(w in lower for w in ["ai", "deepfake", "synthetic", "bot", "algorithm", "neural", "cloned"])
            has_crime = any(w in lower for w in ["fraud", "steal", "stole", "theft", "defrauding", "launder", "embezzl", "phish", "scam"])
            has_defense_or_innocent = any(w in lower for w in ["blocked", "prevented", "defensive", "protect", "legal market", "election"])

            if "ai" in q_lower or "deepfake" in q_lower or "fraud" in q_lower:
                if has_ai and has_crime and not has_defense_or_innocent:
                    decision = "Yes"
                    confidence = 0.98
                    reason = "Context indicates illicit financial loss caused by AI/deepfake manipulation."
                else:
                    decision = "No"
                    confidence = 0.95
                    reason = "Context does not substantiate AI-driven financial crime or is purely defensive/non-financial."
            else:
                decision = "No"
                confidence = 0.90
                reason = "Mock default negative."

        result = {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reason,
            "raw": f'{{"decision": "{decision}", "confidence": {confidence}}}',
            "cached": False,
        }
        self.cache[key] = {**result, "cached": True}
        return result
