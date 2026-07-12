"""Tests for src/query/select.py (SelectDiverse)."""

from __future__ import annotations

from pathlib import Path

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


def test_unreadable_depth_warns_and_drops_only_that_frame(pool_from, meta_db):
    """A corrupt depth PNG must cost one frame's geometry, not the whole query."""
    pool = pool_from(_desc(6))
    Path(pool[0].depth_path).write_bytes(b"not a png")
    step = SelectDiverse(meta_db(), n_diverse=3)

    with pytest.warns(UserWarning, match="cannot read depth for 'id-0'"):
        out = step.invoke(_state(pool))

    assert len(out.retrieved) == 3
    # The other five still had usable geometry, so the selection stayed
    # geometric and had no need to fall back on the unreadable frame.
    assert "id-0" not in {f.id for f in out.retrieved}


def test_a_depth_hole_at_the_frame_centre_invalidates_that_frame(pool_from, meta_db):
    """An all-zero depth patch is a hole, not a 0 m surface — the frame has no
    usable look-at point and must not drag the object centre to the camera."""
    pool = pool_from(_desc(6), depth_values=[0] + [1000] * 5)
    step = SelectDiverse(meta_db(), n_diverse=3)

    out = step.invoke(_state(pool))

    assert len(out.retrieved) == 3
    assert "id-0" not in {f.id for f in out.retrieved}


def test_all_depths_unreadable_falls_back_to_top_n(pool_from, meta_db):
    pool = pool_from(_desc(6))
    for frame in pool:
        Path(frame.depth_path).write_bytes(b"not a png")
    step = SelectDiverse(meta_db(), n_diverse=3)

    with pytest.warns(UserWarning):
        out = step.invoke(_state(pool))

    assert [f.id for f in out.retrieved] == ["id-0", "id-1", "id-2"]


def test_cameras_sitting_on_the_object_fall_back_to_top_n(pool_from, meta_db):
    """Every camera centre coincides with the estimated object position, so no
    viewing direction exists — diversity is undefined and similarity wins.

    Built by parking all six cameras at the origin, half looking +Z and half
    −Z: their look-at points straddle the origin, so the median object centre
    lands exactly on every camera centre and all offsets are zero-length.
    """
    pool = pool_from(_desc(6))
    for i, frame in enumerate(pool):
        pose = np.eye(4, dtype=np.float32)
        if i >= 3:
            pose[:3, :3] = np.diag([1.0, -1.0, -1.0])    # optical axis flipped to −Z
        frame.cam2world = pose
    step = SelectDiverse(meta_db(), n_diverse=3)

    with pytest.warns(UserWarning, match="degenerate"):
        out = step.invoke(_state(pool))

    assert [f.id for f in out.retrieved] == ["id-0", "id-1", "id-2"]


def test_one_camera_on_the_object_is_skipped_not_fatal(pool_from, meta_db):
    """A single degenerate frame drops out of the direction set; the rest still
    get a geometric selection."""
    pool = pool_from(_desc(6))
    pool[0].cam2world = np.eye(4, dtype=np.float32)   # sits at the object centre
    step = SelectDiverse(meta_db(), n_diverse=3)

    out = step.invoke(_state(pool))

    assert len(out.retrieved) == 3


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
