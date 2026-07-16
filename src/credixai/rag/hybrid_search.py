"""Reciprocal Rank Fusion para combinar busqueda densa (Qdrant) y BM25 (paso 5).

RRF combina rankings heterogeneos (embeddings vs. keyword matching) sin
necesitar normalizar sus escalas de score, que no son comparables entre si.
"""


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
