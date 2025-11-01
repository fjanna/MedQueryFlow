from __future__ import annotations

"""Prompt-assisted query rewriter with terminology normalization."""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from modules.llm_client import LLMClient, LLMRequest

logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    rewrites: List[str]
    applied_normalizations: Dict[str, str]


class QueryRewriter:
    def __init__(
        self,
        prompt_path: Path,
        normalization_table: Path,
        llm_client: Optional[LLMClient] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.normalization = self._load_normalization(normalization_table)
        self.llm_client = llm_client or LLMClient()
        self.model = model
        self.temperature = temperature

    @staticmethod
    def _load_normalization(path: Path) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if not path.exists():
            logger.warning("Normalization table missing at %s. Populate it for better rewrites.", path)
            return mapping
        with path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                informal = row.get("informal")
                canonical = row.get("canonical")
                if informal and canonical:
                    mapping[informal.lower()] = canonical
        return mapping

    def _apply_normalization(self, question: str) -> Dict[str, str]:
        applied = {}
        lowered = question.lower()
        for informal, canonical in self.normalization.items():
            if informal in lowered:
                applied[informal] = canonical
        return applied

    def rewrite(self, question: str) -> RewriteResult:
        applied = self._apply_normalization(question)
        prompt = self.prompt_template.format(question=question)
        response = self.llm_client.generate(
            LLMRequest(prompt=prompt, model=self.model, temperature=self.temperature)
        )
        try:
            payload = json.loads(response)
            rewrites = payload.get("rewrites", [])
        except json.JSONDecodeError:
            logger.warning("LLM rewrite JSON decode failed, falling back to heuristic rewrites")
            rewrites = [question]
        # Append canonical variants derived from normalization table.
        for informal, canonical in applied.items():
            if canonical not in rewrites:
                rewrites.append(question.replace(informal, canonical))
        # Guarantee uniqueness and limit to 5 variations.
        deduped: List[str] = []
        for candidate in rewrites:
            normalized = candidate.strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return RewriteResult(rewrites=deduped[:5], applied_normalizations=applied)

