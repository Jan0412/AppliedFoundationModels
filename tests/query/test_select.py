"""Tests for src/query/select.py (SelectDiverse)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data_model import RetrievalDiagnostics, SearchState
from src.query import SelectDiverse
from src.utils.db import connect


def _state(pool, collection_id="coll", mode="dynamic", n_diverse=None) -> SearchState:
    return SearchState(
        query="q",
        collection_id=collection_id,
        n_diverse=n_diverse,
        retrieved=pool,
        retrieval_diag=RetrievalDiagnostics(mode=mode, pool_size=len(pool)),
    )


def _desc(n: int) -> list[float]:
    """n similarity scores, descending — mimics retrieval output order."""
    return np.linspace(0.95, 0.7, n).tolist()


def test_pool_smaller_than_n_keeps_all_and_loads_images(pool_from, meta_db):
    pool = pool_from(_desc(3))
    step = SelectDiverse(meta_db(), n_diverse=10)

    out = step.invoke(_state(pool))

    assert len(out.retrieved) == 3
    assert all(isinstance(f.image, Image.Image) for f in out.retrieved)
    assert out.retrieval_diag.n_selected == 3


def test_selects_viewpoint_spread_frames_on_circle(pool_from, meta_db):
    # 12 cameras every 30° around the object; FPS from the top-similarity
    # frame should pick four ~90°-spread viewpoints: 0°, 90°, 180°, 270°.
    pool = pool_from(_desc(12))
    step = SelectDiverse(meta_db(), n_diverse=4)

    out = step.invoke(_state(pool))

    assert {f.id for f in out.retrieved} == {"id-0", "id-3", "id-6", "id-9"}
    # Seed = highest-similarity frame, and output stays similarity-desc.
    assert out.retrieved[0].id == "id-0"
    sims = [f.similarity_score for f in out.retrieved]
    assert sims == sorted(sims, reverse=True)


def test_selected_loaded_and_pool_rest_stays_unloaded(pool_from, meta_db):
    pool = pool_from(_desc(12))
    out = SelectDiverse(meta_db(), n_diverse=4).invoke(_state(pool))

    selected_ids = {f.id for f in out.retrieved}
    assert all(isinstance(f.image, Image.Image) for f in out.retrieved)
    # The no-eager-load guarantee: unselected pool frames were never opened.
    for frame in pool:
        if frame.id not in selected_ids:
            assert frame.image is None


def test_outlier_lookat_does_not_shift_object_center(pool_from, meta_db):
    # Frame 5's depth claims 50 m — its look-at point lands far outside the
    # scene. The median object centre must ignore it, leaving the selection
    # identical to the clean pool's.
    depths = [1000] * 12
    depths[5] = 50_000
    clean = SelectDiverse(meta_db(), n_diverse=4).invoke(
        _state(pool_from(_desc(12)))
    )
    dirty = SelectDiverse(meta_db(), n_diverse=4).invoke(
        _state(pool_from(_desc(12), depth_values=depths))
    )

    assert {f.id for f in dirty.retrieved} == {f.id for f in clean.retrieved}


def test_invalid_frames_fill_by_similarity_when_valid_short(pool_from, meta_db):
    pool = pool_from(_desc(5))
    pool[1].cam2world = None    # no pose
    pool[2].depth_path = ""     # no depth
    step = SelectDiverse(meta_db(), n_diverse=4)

    out = step.invoke(_state(pool))

    ids = {f.id for f in out.retrieved}
    assert len(ids) == 4
    # All three valid frames picked, then the highest-similarity invalid one.
    assert {"id-0", "id-3", "id-4", "id-1"} == ids


def test_all_invalid_falls_back_to_top_n(pool_from, meta_db):
    pool = pool_from(_desc(6))
    for frame in pool:
        frame.depth_path = ""
    step = SelectDiverse(meta_db(), n_diverse=3)

    with pytest.warns(UserWarning, match="valid depth"):
        out = step.invoke(_state(pool))

    assert [f.id for f in out.retrieved] == ["id-0", "id-1", "id-2"]


def test_missing_collection_meta_falls_back_to_top_n(pool_from, tmp_path):
    pool = pool_from(_desc(6))
    db = connect(tmp_path / "db_without_meta")
    step = SelectDiverse(db, n_diverse=3)

    with pytest.warns(UserWarning, match="no calibration"):
        out = step.invoke(_state(pool))

    assert [f.id for f in out.retrieved] == ["id-0", "id-1", "id-2"]


def test_topk_mode_passes_through_untouched(pool_from, meta_db):
    pool = pool_from(_desc(5))
    state = _state(pool, mode="topk")

    out = SelectDiverse(meta_db(), n_diverse=2).invoke(state)

    assert out is state
    assert len(out.retrieved) == 5
    assert all(f.image is None for f in out.retrieved)


def test_state_n_diverse_overrides_constructor(pool_from, meta_db):
    pool = pool_from(_desc(12))
    out = SelectDiverse(meta_db(), n_diverse=2).invoke(
        _state(pool, n_diverse=3)
    )
    assert len(out.retrieved) == 3


def test_diag_records_selection(pool_from, meta_db):
    pool = pool_from(_desc(12))
    out = SelectDiverse(meta_db(), n_diverse=4).invoke(_state(pool))

    diag = out.retrieval_diag
    assert diag.n_selected == 4
    assert diag.selected_ids == [f.id for f in out.retrieved]


def test_rejects_missing_retrieved(meta_db):
    state = SearchState(query="q", collection_id="coll")
    with pytest.raises(ValueError, match="retrieved"):
        SelectDiverse(meta_db()).invoke(state)
