from __future__ import annotations

"""Safety classifier for triage and compliance highlighting."""

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from modules.intent_classifier import IntentResult
from modules.rag_retriever import RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class SafetyAssessment:
    triage_level: str
    matched_keywords: List[str]
    compliance_flags: List[str]
    escalation_message: Optional[str]
    highlight_message: Optional[str]


class SafetyClassifier:
    """Apply keyword heuristics and configurable gates to detect urgent cases."""

    def __init__(self, safety_config: Dict, triage_default: str = "info") -> None:
        self.config = safety_config or {}
        self.triage_levels: List[str] = self.config.get("triage_levels", ["info", "caution", "urgent"])
        self.priority = {level: idx for idx, level in enumerate(self.triage_levels)}
        self.triage_default = triage_default if triage_default in self.priority else self.triage_levels[0]

        self.urgent_keywords = self._lower(self.config.get("urgent_keywords", []))
        self.caution_keywords = self._lower(self.config.get("caution_keywords", []))
        self.compliance_gates: Dict[str, Dict] = self.config.get("compliance_gates", {})

        self.default_escalation = self.config.get("escalation_prompt")
        self.urgent_highlight = self.config.get(
            "urgent_highlight",
            "Symptoms described here may warrant immediate in-person medical evaluation.",
        )
        self.caution_highlight = self.config.get(
            "caution_highlight",
            "Consider consulting a clinician soon to review concerning symptoms.",
        )

    @staticmethod
    def _lower(values: Iterable[str]) -> List[str]:
        return [value.lower() for value in values]

    def assess(
        self,
        question: str,
        intent: IntentResult,
        documents: Sequence[RetrievedDocument],
    ) -> SafetyAssessment:
        """Return the most severe triage level matched from the configured gates."""

        surface_text = " ".join([question] + [doc.text for doc in documents])
        text_lower = surface_text.lower()

        triage = self.triage_default
        matched_keywords: List[str] = []
        compliance_flags: List[str] = []
        highlight_message: Optional[str] = None
        escalation_message: Optional[str] = None

        def upgrade(level: str, highlight: Optional[str], escalation: Optional[str]) -> None:
            nonlocal triage, highlight_message, escalation_message
            if level not in self.priority:
                logger.debug("Unknown triage level %s; skipping upgrade", level)
                return
            if self.priority[level] >= self.priority[triage]:
                triage = level
                if highlight:
                    highlight_message = highlight
                if escalation:
                    escalation_message = escalation

        urgent_hits = [kw for kw in self.urgent_keywords if kw in text_lower]
        if urgent_hits:
            matched_keywords.extend(urgent_hits)
            upgrade("urgent", self.urgent_highlight, self.default_escalation)

        caution_hits = [kw for kw in self.caution_keywords if kw in text_lower]
        if caution_hits and not urgent_hits:
            matched_keywords.extend([kw for kw in caution_hits if kw not in matched_keywords])
            upgrade("caution", self.caution_highlight, None)

        for gate_name, gate_config in self.compliance_gates.items():
            gate_keywords = self._lower(gate_config.get("keywords", []))
            gate_hits = [kw for kw in gate_keywords if kw in text_lower]
            if not gate_hits:
                continue
            for hit in gate_hits:
                if hit not in matched_keywords:
                    matched_keywords.append(hit)
            compliance_flags.append(f"{gate_name}: {', '.join(gate_hits)}")
            level = gate_config.get("triage_level", "caution")
            highlight = gate_config.get("highlight_message")
            escalation = gate_config.get("escalation_message", self.default_escalation if level == "urgent" else None)
            upgrade(level, highlight, escalation)

        if highlight_message is None:
            if triage == "urgent":
                highlight_message = self.urgent_highlight
                escalation_message = escalation_message or self.default_escalation
            elif triage == "caution":
                highlight_message = self.caution_highlight

        logger.debug("Safety triage %s computed for intent %s", triage, intent.intent)

        return SafetyAssessment(
            triage_level=triage,
            matched_keywords=matched_keywords,
            compliance_flags=compliance_flags,
            escalation_message=escalation_message,
            highlight_message=highlight_message,
        )

