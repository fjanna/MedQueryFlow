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
from modules.safety_classifier import SafetyAssessment, SafetyClassifier

logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    text: str
    safety: SafetyAssessment
    prompt_version: str


class AnswerGenerator:
    def __init__(
        self,
        prompt_path: Path,
        prompt_version: str,
        safety_classifier: SafetyClassifier,
        llm_client: Optional[LLMClient] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ) -> None:
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.prompt_version = prompt_version
        self.safety_classifier = safety_classifier
        self.llm_client = llm_client or LLMClient()
        self.model = model
        self.temperature = temperature

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
        safety = self.safety_classifier.assess(question=question, intent=intent, documents=documents)
        style_hint = {
            "empathetic": "Use warm, validating language and acknowledge feelings.",
            "supportive": "Offer practical tips in a friendly tone.",
            "clinical": "Be structured, list differential possibilities with probabilities if possible.",
            "neutral": "Keep tone objective and concise.",
        }.get(intent.style_profile, "Respond clearly and helpfully.")
        prompt = self.prompt_template.format(context=context, question=question)
        prompt += f"\nTone guidance: {style_hint}\n"
        if safety.highlight_message:
            prompt += f"\nSafety guidance: {safety.highlight_message}\n"
        if safety.triage_level == "urgent":
            prompt += "\nIMPORTANT: Provide clear escalation advice for urgent warning signs.\n"
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

