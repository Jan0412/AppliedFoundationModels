"""Tests for src/query/retrieve.py (topk + dynamic modes)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data_model import SearchState
from src.query import RetrieveSimilar


def _state_with_query_vec(populated_db, vec: np.ndarray, *, top_k: int = 5) -> SearchState:
    return SearchState(
        query="anything",
        collection_id=populated_db["collection_id"],
        top_k_retrieve=top_k,
        query_embedding=vec,
    )


def _unit_query(embed_dim: int) -> np.ndarray:
    q = np.zeros(embed_dim, dtype=np.float32)
    q[0] = 1.0
    return q


def _dyn_state(dyn, embed_dim: int, **kwargs) -> SearchState:
    return SearchState(
        query="anything",
        collection_id=dyn["collection_id"],
        query_embedding=_unit_query(embed_dim),
        **kwargs,
    )


BIMODAL_SIMS = np.concatenate([
    np.linspace(0.85, 0.95, 12),    # relevant cluster
    np.linspace(0.05, 0.15, 48),    # background
]).tolist()


# ---------------------------------------------------------------------------
# topk mode (legacy behavior)
# ---------------------------------------------------------------------------


def test_invoke_returns_topk_ordered_by_similarity(populated_db, embed_dim):
    step = RetrieveSimilar(populated_db["db"])
    # Query exactly matches the first stored vector.
    state = _state_with_query_vec(populated_db, _unit_query(embed_dim), top_k=3)

    out = step.invoke(state)

    assert out.retrieved is not None
    assert len(out.retrieved) == 3
    # First hit should be id-0 (exact match → similarity == 1.0).
    assert out.retrieved[0].id == "id-0"
    assert out.retrieved[0].similarity_score == pytest.approx(1.0, abs=1e-5)
    # Subsequent hits are orthogonal → similarity == 0.0.
    for hit in out.retrieved[1:]:
        assert hit.similarity_score == pytest.approx(0.0, abs=1e-5)


def test_invoke_defers_image_loading(populated_db, embed_dim):
    step = RetrieveSimilar(populated_db["db"])
    state = _state_with_query_vec(populated_db, _unit_query(embed_dim), top_k=3)

    out = step.invoke(state)

    for hit in out.retrieved:
        assert hit.image is None
        loaded = hit.load_image()
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
        assert hit.image is loaded  # cached


def test_invoke_respects_top_k_retrieve(populated_db, embed_dim):
    step = RetrieveSimilar(populated_db["db"])
    state = _state_with_query_vec(populated_db, _unit_query(embed_dim), top_k=2)

    out = step.invoke(state)

    assert len(out.retrieved) == 2


def test_topk_writes_diagnostics(populated_db, embed_dim):
    step = RetrieveSimilar(populated_db["db"])
    state = _state_with_query_vec(populated_db, _unit_query(embed_dim), top_k=3)

    out = step.invoke(state)

    assert out.retrieval_diag is not None
    assert out.retrieval_diag.mode == "topk"
    assert out.retrieval_diag.pool_size == 3


def test_invoke_rejects_missing_query_embedding(populated_db):
    step = RetrieveSimilar(populated_db["db"])
    state = SearchState(query="x", collection_id="coll")  # no embedding

    with pytest.raises(ValueError, match="query_embedding"):
        step.invoke(state)


def test_rejects_unknown_mode_and_strategy(populated_db):
    with pytest.raises(ValueError, match="mode"):
        RetrieveSimilar(populated_db["db"], mode="bogus")
    with pytest.raises(ValueError, match="strategy"):
        RetrieveSimilar(populated_db["db"], strategy="bogus")


# ---------------------------------------------------------------------------
# dynamic mode
# ---------------------------------------------------------------------------


def test_dynamic_pool_keeps_the_relevant_cluster(db_with_sims, embed_dim):
    dyn = db_with_sims(BIMODAL_SIMS)
    step = RetrieveSimilar(
        dyn["db"], mode="dynamic", strategy="plain", min_k=2, max_k=100
    )

    out = step.invoke(_dyn_state(dyn, embed_dim))

    diag = out.retrieval_diag
    assert diag.mode == "dynamic"
    assert diag.gated is False
    assert diag.total_frames == 60
    # Plain Otsu splits in the gap → keeps exactly the 12-score cluster.
    assert len(out.retrieved) == 12
    assert diag.pool_size == 12
    # σ²_B is flat across the gap → threshold sits just above the background
    # maximum (0.15); anything in the open gap (0.15, 0.85) is a correct split.
    assert 0.15 < diag.threshold < 0.85
    sims = [h.similarity_score for h in out.retrieved]
    assert sims == sorted(sims, reverse=True)
    assert min(sims) > 0.8
    assert all(h.image is None for h in out.retrieved)


def test_dynamic_tail_strategy_trims_within_cluster(db_with_sims, embed_dim):
    dyn = db_with_sims(BIMODAL_SIMS, collection_id="dyntail")
    step = RetrieveSimilar(
        dyn["db"], mode="dynamic", strategy="tail", min_k=2, max_k=100
    )

    out = step.invoke(_dyn_state(dyn, embed_dim))

    assert out.retrieval_diag.gated is False
    assert 2 <= len(out.retrieved) <= 12
    assert out.retrieval_diag.threshold >= 0.2


def test_dynamic_gates_on_unimodal_scores(db_with_sims, embed_dim):
    rng = np.random.default_rng(2)
    sims = np.clip(rng.normal(0.5, 0.03, 60), 0.0, 1.0).tolist()
    dyn = db_with_sims(sims, collection_id="dynuni")
    step = RetrieveSimilar(dyn["db"], mode="dynamic", min_k=4, max_k=100)

    out = step.invoke(_dyn_state(dyn, embed_dim))

    assert out.retrieval_diag.gated is True
    assert len(out.retrieved) == 4  # top-min_k fallback


def test_dynamic_max_k_caps_pool(db_with_sims, embed_dim):
    dyn = db_with_sims(BIMODAL_SIMS, collection_id="dyncap")
    step = RetrieveSimilar(
        dyn["db"], mode="dynamic", strategy="plain", min_k=2, max_k=5
    )

    out = step.invoke(_dyn_state(dyn, embed_dim))

    assert len(out.retrieved) == 5


def test_state_retrieval_mode_overrides_constructor(db_with_sims, embed_dim):
    dyn = db_with_sims(BIMODAL_SIMS, collection_id="dynovr")

    # topk constructor, dynamic per query:
    step = RetrieveSimilar(dyn["db"], mode="topk", strategy="plain", min_k=2)
    out = step.invoke(_dyn_state(dyn, embed_dim, retrieval_mode="dynamic"))
    assert out.retrieval_diag.mode == "dynamic"
    assert len(out.retrieved) == 12

    # dynamic constructor, topk per query:
    step = RetrieveSimilar(dyn["db"], mode="dynamic")
    out = step.invoke(
        _dyn_state(dyn, embed_dim, retrieval_mode="topk", top_k_retrieve=7)
    )
    assert out.retrieval_diag.mode == "topk"
    assert len(out.retrieved) == 7


def test_dynamic_empty_collection(db_with_sims, embed_dim):
    dyn = db_with_sims([], collection_id="dynempty")
    step = RetrieveSimilar(dyn["db"], mode="dynamic")

    out = step.invoke(_dyn_state(dyn, embed_dim))

    assert out.retrieved == []
    assert out.retrieval_diag.gated is True
    assert out.retrieval_diag.total_frames == 0
