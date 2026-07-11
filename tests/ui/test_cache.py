"""Tests for src.ui.cache.SceneCloudCache."""

from __future__ import annotations

import numpy as np
import pytest

import src.ui.cache as cache_mod
from src.ui.cache import SceneCloudCache


@pytest.fixture
def counting_build(monkeypatch):
    """Replace build_scene_cloud with a counter returning a fixed cloud."""
    calls = {"n": 0}
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    colors = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)

    def _fake(fs, n_frames, voxel):
        calls["n"] += 1
        return points, colors

    monkeypatch.setattr(cache_mod, "build_scene_cloud", _fake)
    return calls, points, colors


def test_miss_builds_and_hit_reuses(tmp_path, populated_store, counting_build):
    db, cid = populated_store
    calls, points, colors = counting_build
    cache = SceneCloudCache(tmp_path / "cache")

    p1, c1 = cache.get(db, cid)
    p2, c2 = cache.get(db, cid)

    assert calls["n"] == 1                       # second call served from disk
    np.testing.assert_array_equal(p1, points)
    np.testing.assert_array_equal(c1, colors)
    np.testing.assert_array_equal(p2, points)
    np.testing.assert_array_equal(c2, colors)


def test_force_rebuilds(tmp_path, populated_store, counting_build):
    db, cid = populated_store
    calls, *_ = counting_build
    cache = SceneCloudCache(tmp_path / "cache")

    cache.get(db, cid)
    cache.get(db, cid, force=True)
    assert calls["n"] == 2


def test_changed_voxel_invalidates(tmp_path, populated_store, counting_build):
    db, cid = populated_store
    calls, *_ = counting_build

    SceneCloudCache(tmp_path / "cache", voxel=0.02).get(db, cid)
    # A different voxel would produce a different cloud, so the file can't be reused.
    SceneCloudCache(tmp_path / "cache", voxel=0.05).get(db, cid)
    assert calls["n"] == 2


def test_changed_row_count_invalidates(tmp_path, populated_store, counting_build):
    """Re-ingesting a collection (row count changes) rebuilds without asking."""
    db, cid = populated_store
    calls, *_ = counting_build
    cache = SceneCloudCache(tmp_path / "cache")

    cache.get(db, cid)
    db.open_table(cid).add([{
        "id": "id-extra", "collection_id": cid,
        "vector": np.eye(1, 8, dtype=np.float32)[0].tolist(),
        "path": "x.png", "depth_path": "x.png",
        "cam2world": np.eye(4, dtype=np.float32).reshape(16).tolist(),
    }])
    cache.get(db, cid)
    assert calls["n"] == 2


def test_invalidate_removes_the_file(tmp_path, populated_store, counting_build):
    db, cid = populated_store
    calls, *_ = counting_build
    cache = SceneCloudCache(tmp_path / "cache")

    cache.get(db, cid)
    assert cache.path_for(cid).is_file()
    cache.invalidate(cid)
    assert not cache.path_for(cid).is_file()
    cache.get(db, cid)                            # miss again
    assert calls["n"] == 2


def test_colors_none_round_trips(tmp_path, populated_store, monkeypatch):
    db, cid = populated_store
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    monkeypatch.setattr(cache_mod, "build_scene_cloud", lambda fs, n_frames, voxel: (points, None))
    cache = SceneCloudCache(tmp_path / "cache")

    cache.get(db, cid)
    p, c = cache.get(db, cid)                     # from disk
    assert c is None
    np.testing.assert_array_equal(p, points)


def test_unsafe_collection_id_is_rejected(tmp_path):
    cache = SceneCloudCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="unsafe"):
        cache.path_for("../etc/passwd")
