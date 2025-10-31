from __future__ import annotations

"""Prompt-oriented intent classifier with rule-based fallback."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from modules.llm_client import LLMClient, LLMRequest

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    intent: str
    confidence: float
    style_profile: str
    routing_target: str


class IntentClassifier:
    def __init__(
        self,
        prompt_path: Path,
        routing_rules: Dict[str, str],
        llm_client: Optional[LLMClient] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.routing_rules = routing_rules
        self.llm_client = llm_client or LLMClient()
        self.model = model
        self.temperature = temperature

    @staticmethod
    def _heuristic_intent(question: str) -> IntentResult:
        q = question.lower()
        if any(k in q for k in ["feel", "worried", "scared", "anxious", "embarrassed"]):
            return IntentResult("emotional_support", 0.55, "empathetic", "in_app_notes")
        if re.search(r"\b(can|should) i\b", q) or "好吗" in q:
            return IntentResult("care_advice", 0.6, "supportive", "external_corpus")
        if any(k in q for k in ["symptom", "diagnos", "是什么", "可能是什么"]):
            return IntentResult("diagnostic_request", 0.65, "clinical", "external_corpus")
        if any(k in q for k in ["什么是", "定义", "机制", "how does"]):
            return IntentResult("medical_fact", 0.6, "neutral", "external_corpus")
        return IntentResult("other", 0.4, "neutral", "external_corpus")

    def classify(self, question: str) -> IntentResult:
        prompt = self.prompt_template.format(question=question)
        response = self.llm_client.generate(
            LLMRequest(prompt=prompt, model=self.model, temperature=self.temperature)
        )
        try:
            payload = json.loads(response)
            intent = payload.get("intent")
            confidence = float(payload.get("confidence", 0.5))
            logger.debug("LLM intent payload: %s", payload)
        except json.JSONDecodeError:
            logger.warning("Falling back to heuristic intent due to JSON decode error")
            return self._heuristic_intent(question)

        heuristic = self._heuristic_intent(question)
        style_profile = {
            "medical_fact": "neutral",
            "diagnostic_request": "clinical",
            "care_advice": "supportive",
            "emotional_support": "empathetic",
        }.get(intent, heuristic.style_profile)
        routing_target = self.routing_rules.get(intent, heuristic.routing_target)
        return IntentResult(intent=intent, confidence=confidence, style_profile=style_profile, routing_target=routing_target)

