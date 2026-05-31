"""Tests for src/query/retrieve.py."""

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


def test_invoke_returns_topk_ordered_by_similarity(populated_db, embed_dim):
    step = RetrieveSimilar(populated_db["db"])
    # Query exactly matches the first stored vector.
    q = np.zeros(embed_dim, dtype=np.float32)
    q[0] = 1.0
    state = _state_with_query_vec(populated_db, q, top_k=3)

    out = step.invoke(state)

    assert out.retrieved is not None
    assert len(out.retrieved) == 3
    # First hit should be id-0 (exact match → similarity == 1.0).
    assert out.retrieved[0].id == "id-0"
    assert out.retrieved[0].similarity_score == pytest.approx(1.0, abs=1e-5)
    # Subsequent hits are orthogonal → similarity == 0.0.
    for hit in out.retrieved[1:]:
        assert hit.similarity_score == pytest.approx(0.0, abs=1e-5)


def test_invoke_loads_pil_image_for_each_hit(populated_db, embed_dim):
    step = RetrieveSimilar(populated_db["db"])
    q = np.zeros(embed_dim, dtype=np.float32)
    q[0] = 1.0
    state = _state_with_query_vec(populated_db, q, top_k=3)

    out = step.invoke(state)

    for hit in out.retrieved:
        assert isinstance(hit.image, Image.Image)
        assert hit.image.mode == "RGB"


def test_invoke_respects_top_k_retrieve(populated_db, embed_dim):
    step = RetrieveSimilar(populated_db["db"])
    q = np.zeros(embed_dim, dtype=np.float32)
    q[0] = 1.0
    state = _state_with_query_vec(populated_db, q, top_k=2)

    out = step.invoke(state)

    assert len(out.retrieved) == 2


def test_invoke_rejects_missing_query_embedding(populated_db):
    step = RetrieveSimilar(populated_db["db"])
    state = SearchState(query="x", collection_id="coll")  # no embedding

    with pytest.raises(ValueError, match="query_embedding"):
        step.invoke(state)
