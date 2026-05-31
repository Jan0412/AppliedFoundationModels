"""Tests for src/query/embed.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_model import SearchState
from src.query import EmbedQuery


def test_invoke_sets_query_embedding(mock_siglip_model, embed_dim):
    step = EmbedQuery(mock_siglip_model)
    state = SearchState(query="a cat", collection_id="coll")

    out = step.invoke(state)

    assert isinstance(out.query_embedding, np.ndarray)
    assert out.query_embedding.shape == (embed_dim,)
    mock_siglip_model.embed_text.assert_called_once_with("a cat")


def test_invoke_does_not_mutate_input_state(mock_siglip_model):
    step = EmbedQuery(mock_siglip_model)
    state = SearchState(query="a cat", collection_id="coll")

    out = step.invoke(state)

    assert state.query_embedding is None  # original untouched
    assert out is not state                 # new instance
    # Other fields preserved
    assert out.query == state.query
    assert out.collection_id == state.collection_id
    assert out.top_k_retrieve == state.top_k_retrieve
    assert out.top_k_final == state.top_k_final


def test_invoke_rejects_empty_query(mock_siglip_model):
    step = EmbedQuery(mock_siglip_model)
    state = SearchState(query="", collection_id="coll")

    with pytest.raises(ValueError, match="state.query"):
        step.invoke(state)
