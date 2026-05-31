"""Tests for src/query/rerank.py."""

from __future__ import annotations

import pytest

from src.data_model import SearchState
from src.query import RerankByDetection


def test_invoke_sorts_descending_and_trims(detected_factory):
    detected = detected_factory([0.1, 0.9, 0.4, 0.7])
    state = SearchState(
        query="q",
        collection_id="c",
        top_k_final=2,
        detected=detected,
    )

    out = RerankByDetection().invoke(state)

    assert out.results is not None
    assert len(out.results) == 2
    assert [r.detection_score for r in out.results] == [0.9, 0.7]


def test_invoke_handles_top_k_larger_than_input(detected_factory):
    detected = detected_factory([0.3, 0.5])
    state = SearchState(
        query="q",
        collection_id="c",
        top_k_final=10,
        detected=detected,
    )

    out = RerankByDetection().invoke(state)

    assert [r.detection_score for r in out.results] == [0.5, 0.3]


def test_invoke_does_not_mutate_detected(detected_factory):
    detected = detected_factory([0.1, 0.9, 0.4])
    state = SearchState(
        query="q",
        collection_id="c",
        top_k_final=2,
        detected=detected,
    )

    out = RerankByDetection().invoke(state)

    # state.detected preserved in input order on the *original* state.
    assert [d.detection_score for d in state.detected] == [0.1, 0.9, 0.4]
    # Output state still references the full detected list.
    assert [d.detection_score for d in out.detected] == [0.1, 0.9, 0.4]


def test_invoke_rejects_missing_detected():
    state = SearchState(query="q", collection_id="c")
    with pytest.raises(ValueError, match="detected"):
        RerankByDetection().invoke(state)
