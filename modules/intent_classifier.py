from __future__ import annotations

"""Prompt-oriented intent classifier that delegates fully to the LLM."""

import json
import logging
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
        self.default_style_profile = "neutral"
        self.style_profiles = {
            "medical_fact": "neutral",
            "diagnostic_request": "clinical",
            "care_advice": "supportive",
            "emotional_support": "empathetic",
        }
        self.default_routing_target = routing_rules.get("other", "external_corpus")

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
            logger.error("Unable to parse LLM intent response; defaulting to 'other'")
            payload = {}
            intent = None
            confidence = 0.0

        if not intent:
            logger.error("LLM intent response missing 'intent'; defaulting to 'other'")
            intent = "other"

        style_profile = payload.get(
            "style_profile",
            self.style_profiles.get(intent, self.default_style_profile),
        )
        routing_target = payload.get(
            "routing_target",
            self.routing_rules.get(intent, self.default_routing_target),
        )
        return IntentResult(
            intent=intent,
            confidence=confidence,
            style_profile=style_profile,
            routing_target=routing_target,
        )

