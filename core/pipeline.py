from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from modules.answer_generator import AnswerGenerator
from modules.intent_classifier import IntentClassifier, IntentResult
from modules.query_rewriter import QueryRewriter, RewriteResult
from modules.rag_retriever import RAGRetriever, RetrievedDocument
from modules.llm_client import LLMClient
from utils.exporter import export_run

logger = logging.getLogger(__name__)


@dataclass
class PipelineVariant:
    name: str
    prompt_version: str
    index_version: str
    safety_ruleset_version: str


class MedQueryFlowPipeline:
    def __init__(self, config_path: Path, llm_client: Optional[LLMClient] = None) -> None:
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.config_path = config_path
        self.llm_client = llm_client or LLMClient()
        self.variants = [PipelineVariant(**variant) for variant in self.config["app"]["ab_testing"]["variants"]]
        self.prompts = self.config["paths"]["prompts"]
        self.intent_classifier = IntentClassifier(
            prompt_path=Path(self.prompts["intent"]),
            routing_rules=self.config.get("routing", {}),
            llm_client=self.llm_client,
            model=self.config["llm"]["model_map"]["intent"],
            temperature=self.config["llm"]["temperature"]["intent"],
        )
        self.query_rewriter = QueryRewriter(
            prompt_path=Path(self.prompts["rewrite"]),
            normalization_table=Path(self.config["paths"]["normalization_table"]),
            llm_client=self.llm_client,
            model=self.config["llm"]["model_map"]["rewrite"],
            temperature=self.config["llm"]["temperature"]["rewrite"],
        )
        self.retrievers: Dict[str, RAGRetriever] = {}
        self.answer_generators: Dict[str, AnswerGenerator] = {}

    def _select_variant(self, name: Optional[str]) -> PipelineVariant:
        if name:
            for variant in self.variants:
                if variant.name == name:
                    return variant
            logger.warning("Variant %s not found. Falling back to default.", name)
        return self.variants[0]

    def run(
        self,
        question: str,
        variant_name: Optional[str] = None,
        enable_rewrites: bool = True,
        export: bool = False,
    ) -> Dict[str, Any]:
        variant = self._select_variant(variant_name)

        intent: IntentResult = self.intent_classifier.classify(question)
        rewrite_result: RewriteResult
        if enable_rewrites and intent.intent != "emotional_support":
            rewrite_result = self.query_rewriter.rewrite(question)
            retrieval_queries = [question] + rewrite_result.rewrites
        else:
            rewrite_result = RewriteResult(rewrites=[question], applied_normalizations={})
            retrieval_queries = [question]

        retriever = self._get_retriever(variant.index_version)
        documents: List[RetrievedDocument] = retriever.retrieve(retrieval_queries)
        answer_generator = self._get_answer_generator(variant.prompt_version)
        answer = answer_generator.generate(
            question=question,
            intent=intent,
            documents=documents,
            safety_ruleset_version=variant.safety_ruleset_version,
        )

        run_payload = {
            "question": question,
            "variant": variant.name,
            "intent": intent.__dict__,
            "rewrites": rewrite_result.rewrites,
            "applied_normalizations": rewrite_result.applied_normalizations,
            "retrieved_documents": [doc.__dict__ for doc in documents],
            "answer": answer.text,
            "safety": answer.safety.__dict__,
        }

        if export:
            export_path = export_run(
                export_dir=self.config["logging"]["export_dir"],
                filename_template=self.config["logging"]["export_filename_template"],
                payload=run_payload,
                prompt_version=answer.prompt_version,
                index_version=variant.index_version,
                safety_ruleset_version=variant.safety_ruleset_version,
            )
            run_payload["export_path"] = str(export_path)

        return run_payload

    def _get_retriever(self, index_version: str) -> RAGRetriever:
        if index_version not in self.retrievers:
            self.retrievers[index_version] = RAGRetriever(
                knowledge_base_path=Path(self.config["paths"]["knowledge_base"]),
                index_version=index_version,
            )
        return self.retrievers[index_version]

    def _get_answer_generator(self, prompt_version: str) -> AnswerGenerator:
        if prompt_version not in self.answer_generators:
            self.answer_generators[prompt_version] = AnswerGenerator(
                prompt_path=Path(self.prompts["answer"]),
                prompt_version=prompt_version,
                safety_config=self.config.get("safety", {}),
                llm_client=self.llm_client,
                model=self.config["llm"]["model_map"]["answer"],
                temperature=self.config["llm"]["temperature"]["answer"],
            )
        return self.answer_generators[prompt_version]

    def load_samples(self) -> List[Dict[str, Any]]:
        sample_path = Path(self.config["app"].get("sample_runs_path", ""))
        if sample_path.exists():
            return json.loads(sample_path.read_text(encoding="utf-8"))
        return []

