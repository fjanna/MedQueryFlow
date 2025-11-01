from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return hits / float(len(relevant_set))


def ndcg_at_k(scores: Sequence[float], k: int) -> float:
    import math

    gains = scores[:k]
    if not gains:
        return 0.0
    dcg = sum((gain / math.log2(idx + 2)) for idx, gain in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum((gain / math.log2(idx + 2)) for idx, gain in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def pairwise_compare(a_scores: List[float], b_scores: List[float]) -> List[Tuple[int, float]]:
    return [(idx, b - a) for idx, (a, b) in enumerate(zip(a_scores, b_scores))]

