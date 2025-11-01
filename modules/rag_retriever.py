from __future__ import annotations

"""Simple TF-IDF based retriever to keep the demo self-contained."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    doc_id: str
    title: str
    text: str
    score: float


class RAGRetriever:
    def __init__(self, knowledge_base_path: Path, index_version: str) -> None:
        self.knowledge_base_path = knowledge_base_path
        self.index_version = index_version
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.documents: List[RetrievedDocument] = []
        self.matrix = None
        self._load_corpus()

    def _load_corpus(self) -> None:
        if not self.knowledge_base_path.exists():
            logger.warning(
                "Knowledge base not found at %s. Add JSON documents with fields id/title/abstract.",
                self.knowledge_base_path,
            )
            return
        with self.knowledge_base_path.open(encoding="utf-8") as handle:
            raw_docs = json.load(handle)
        corpus_texts = []
        self.documents = []
        for entry in raw_docs:
            doc = RetrievedDocument(
                doc_id=str(entry.get("id")),
                title=entry.get("title", "Untitled"),
                text=entry.get("abstract", ""),
                score=0.0,
            )
            self.documents.append(doc)
            corpus_texts.append(doc.text)
        if corpus_texts:
            self.matrix = self.vectorizer.fit_transform(corpus_texts)
            logger.info("Loaded %d documents into index %s", len(self.documents), self.index_version)
        else:
            logger.warning("Knowledge base is empty. Populate %s with medical abstracts.", self.knowledge_base_path)

    def retrieve(self, queries: Sequence[str], top_k: int = 3) -> List[RetrievedDocument]:
        if not self.documents or self.matrix is None:
            logger.warning("Retriever called before documents were loaded.")
            return []
        query_matrix = self.vectorizer.transform(queries)
        scores = cosine_similarity(query_matrix, self.matrix)
        combined_scores = scores.max(axis=0)
        ranked_indices = combined_scores.argsort()[::-1][:top_k]
        results: List[RetrievedDocument] = []
        for idx in ranked_indices:
            doc = self.documents[int(idx)]
            results.append(
                RetrievedDocument(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.text,
                    score=float(combined_scores[idx]),
                )
            )
        return results

