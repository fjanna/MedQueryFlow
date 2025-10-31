from __future__ import annotations

"""Answer generator combining RAG context, safety rules, and prompt control."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from modules.intent_classifier import IntentResult
from modules.llm_client import LLMClient, LLMRequest
from modules.rag_retriever import RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheck:
    triage_level: str
    matched_keywords: List[str]
    escalation_message: Optional[str] = None


@dataclass
class AnswerResult:
    text: str
    safety: SafetyCheck
    prompt_version: str


class AnswerGenerator:
    def __init__(
        self,
        prompt_path: Path,
        prompt_version: str,
        safety_config: dict,
        llm_client: Optional[LLMClient] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ) -> None:
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.prompt_version = prompt_version
        self.safety_config = safety_config
        self.llm_client = llm_client or LLMClient()
        self.model = model
        self.temperature = temperature

    def _run_safety(self, question: str) -> SafetyCheck:
        urgent_keywords = [kw.lower() for kw in self.safety_config.get("urgent_keywords", [])]
        matches = [kw for kw in urgent_keywords if kw in question.lower()]
        level = "info"
        escalation = None
        if matches:
            level = "urgent"
            escalation = self.safety_config.get("escalation_prompt")
        return SafetyCheck(triage_level=level, matched_keywords=matches, escalation_message=escalation)

    @staticmethod
    def _format_context(docs: List[RetrievedDocument]) -> str:
        formatted = []
        for doc in docs:
            formatted.append(f"- ({doc.doc_id}) {doc.title}: {doc.text}")
        return "\n".join(formatted)

    def generate(
        self,
        question: str,
        intent: IntentResult,
        documents: List[RetrievedDocument],
        safety_ruleset_version: str,
    ) -> AnswerResult:
        context = self._format_context(documents)
        safety = self._run_safety(question)
        style_hint = {
            "empathetic": "Use warm, validating language and acknowledge feelings.",
            "supportive": "Offer practical tips in a friendly tone.",
            "clinical": "Be structured, list differential possibilities with probabilities if possible.",
            "neutral": "Keep tone objective and concise.",
        }.get(intent.style_profile, "Respond clearly and helpfully.")
        prompt = self.prompt_template.format(context=context, question=question)
        prompt += f"\nTone guidance: {style_hint}\n"
        if safety.triage_level == "urgent":
            prompt += "\nIMPORTANT: Emphasize that immediate medical attention is required.\n"
        response = self.llm_client.generate(
            LLMRequest(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                metadata={
                    "prompt_version": self.prompt_version,
                    "safety_ruleset_version": safety_ruleset_version,
                },
            )
        )
        if isinstance(response, str) and response.strip().startswith("{"):
            try:
                payload = json.loads(response)
                answer_text = payload.get("answer", "")
            except json.JSONDecodeError:
                answer_text = response
        else:
            answer_text = response
        return AnswerResult(text=answer_text, safety=safety, prompt_version=self.prompt_version)

