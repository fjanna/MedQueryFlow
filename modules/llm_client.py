"""Utility wrapper for LLM calls with pluggable backend.

This file contains a simple synchronous client interface that can be swapped for
real API calls (e.g., OpenAI, Anthropic). For the demo we provide a
rule-based fallback to avoid network calls. Replace `RuleBasedLLM` with your
API implementation and wire credentials via environment variables.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Container describing an LLM prompt invocation."""

    prompt: str
    model: str
    temperature: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class RuleBasedLLM:
    """Deterministic fallback that approximates expected JSON outputs.

    It keeps the project runnable without external API keys. Swap this class
    for a real API client when integrating with your prompt management stack.
    """

    def complete(self, request: LLMRequest) -> str:
        prompt_lower = request.prompt.lower()
        if "classify" in prompt_lower and "intent" in prompt_lower:
            return json.dumps({"intent": "medical_fact", "confidence": 0.51})
        if "rewrite" in prompt_lower:
            return json.dumps(
                {
                    "rewrites": [
                        "What are common medical explanations for chest tightness?",
                        "Could digestive issues cause a feeling of chest discomfort?",
                        "How do clinicians differentiate cardiac and gastric causes of chest tightness?",
                    ]
                }
            )
        # Default answer output referencing the prompt directly.
        return (
            "Based on the supplied context, monitor symptoms, seek care if they worsen, "
            "and consult a licensed clinician for personalized evaluation."
        )


class LLMClient:
    """High-level LLM client with dependency injection."""

    def __init__(self, backend: Optional[RuleBasedLLM] = None) -> None:
        self.backend = backend or RuleBasedLLM()

    def generate(self, request: LLMRequest) -> str:
        logger.debug("Dispatching LLM request: %s", request)
        response = self.backend.complete(request)
        logger.debug("LLM response: %s", response)
        return response

