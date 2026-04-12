from __future__ import annotations

from stage2_ingestion.embedding_benchmark import compute_query_metrics


def test_compute_query_metrics_basic_case() -> None:
    ranked = ["e2", "e1", "e3", "e4"]
    relevant = {"e1", "e3"}
    metrics = compute_query_metrics(ranked, relevant, ks=[1, 2, 3])

    assert metrics["precision@1"] == 0.0
    assert metrics["recall@1"] == 0.0
    assert metrics["hit@1"] == 0.0
    assert metrics["mrr@1"] == 0.0

    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 0.5
    assert metrics["hit@2"] == 1.0
    assert metrics["mrr@2"] == 0.5

    assert metrics["precision@3"] == 2 / 3
    assert metrics["recall@3"] == 1.0
    assert metrics["hit@3"] == 1.0
    assert metrics["mrr@3"] == 0.5
    assert 0.0 <= metrics["ndcg@3"] <= 1.0


def test_compute_query_metrics_raises_on_empty_relevant() -> None:
    try:
        compute_query_metrics(["e1"], set(), ks=[1])
    except ValueError as exc:
        assert "relevant_entity_ids" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty relevant set")


def test_compute_query_metrics_raises_on_invalid_k() -> None:
    try:
        compute_query_metrics(["e1"], {"e1"}, ks=[0])
    except ValueError as exc:
        assert "positive integers" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid ks")
