"""Tests para credixai.rag.hybrid_search (paso 5, prd.md 9.1).

TDD: se escribe antes que credixai/rag/hybrid_search.py. Reciprocal Rank
Fusion es logica pura sobre listas de ids rankeados, sin embeddings ni
llamadas a red: se testea con ids sinteticos.
"""

from credixai.rag.hybrid_search import reciprocal_rank_fusion


def test_rrf_item_ranked_first_in_both_lists_wins():
    dense = ["a", "b", "c"]
    sparse = ["a", "c", "b"]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert fused[0][0] == "a"


def test_rrf_returns_scores_sorted_descending():
    dense = ["a", "b", "c"]
    sparse = ["c", "b", "a"]

    fused = reciprocal_rank_fusion([dense, sparse])

    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_item_present_in_only_one_list_still_included():
    dense = ["a", "b"]
    sparse = ["c"]

    fused = reciprocal_rank_fusion([dense, sparse])

    assert {item for item, _ in fused} == {"a", "b", "c"}


def test_rrf_item_in_both_lists_scores_higher_than_item_in_one():
    dense = ["a", "b"]
    sparse = ["a", "c"]

    fused = reciprocal_rank_fusion([dense, sparse])

    scores = dict(fused)
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_rrf_empty_rankings_returns_empty_list():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_k_parameter_controls_how_much_top_ranks_dominate():
    dense = ["a", "b", "c", "d", "e"]
    sparse = ["e", "d", "c", "b", "a"]

    fused_low_k = dict(reciprocal_rank_fusion([dense, sparse], k=1))
    fused_high_k = dict(reciprocal_rank_fusion([dense, sparse], k=1000))

    # con k chico, la ventaja del primer puesto pesa mucho mas que con k grande
    gap_low_k = fused_low_k["a"] - fused_low_k["c"]
    gap_high_k = fused_high_k["a"] - fused_high_k["c"]
    assert gap_low_k > gap_high_k
