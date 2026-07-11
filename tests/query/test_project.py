"""Tests for src.query.project.ProjectTo3D (multi-frame fusion → one box)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data_model import DetectedImage, SearchState
from src.query.project import ProjectTo3D


def _detected(populated_db, *, frame=0, mask=None, masks_none=False, boxes=None,
              cam2world=None):
    """Build a DetectedImage wired to one frame of the populated_db fixture."""
    if mask is None:
        mask = torch.ones(8, 8, dtype=torch.bool)
    if boxes is None:
        boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0]])
    if cam2world is None:
        cam2world = np.eye(4, dtype=np.float32)
    return DetectedImage(
        id=populated_db["ids"][frame],
        path=populated_db["paths"][frame],
        similarity_score=1.0,
        detection_score=0.9,
        boxes=boxes,
        scores=torch.tensor([0.9] * len(boxes)),
        masks=None if masks_none else [mask],
        depth_path=populated_db["depth_paths"][frame],
        cam2world=cam2world,
    )


def _projector(populated_db, **kw):
    # Permissive clustering: the sparse synthetic points (0.25 m apart) form a
    # single real cluster, exercising the cluster path rather than the fallback.
    params = dict(voxel=0.0, cluster_eps=1.0, cluster_min_samples=1)
    params.update(kw)
    return ProjectTo3D(populated_db["db"], **params)


def _state(populated_db, results):
    return SearchState(
        query="x", collection_id=populated_db["collection_id"], results=results
    )


def test_fuses_mask_result_into_single_object(populated_db):
    di = _detected(populated_db)   # full 8x8 mask, depth=1m everywhere
    out = _projector(populated_db).invoke(_state(populated_db, [di]))

    assert out.projected is not None and len(out.projected) == 1
    obj = out.projected[0]
    assert obj.id == "fused"
    assert obj.points.shape == (64, 3)        # all 64 pixels valid
    assert obj.colors is not None and obj.colors.shape == (64, 3)
    assert obj.bbox is not None and obj.bbox.shape == (2, 3)
    assert np.all(obj.bbox[1] >= obj.bbox[0])


def test_fuses_multiple_frames_into_one_object(populated_db):
    hits = [_detected(populated_db, frame=0), _detected(populated_db, frame=1)]
    out = _projector(populated_db).invoke(_state(populated_db, hits))
    assert len(out.projected) == 1            # still a single fused object
    assert out.projected[0].points.shape[0] >= 1


def test_falls_back_to_detected_when_no_results(populated_db):
    di = _detected(populated_db)
    state = SearchState(
        query="x", collection_id=populated_db["collection_id"], detected=[di]
    )
    out = _projector(populated_db).invoke(state)
    assert len(out.projected) == 1


def test_missing_calibration_raises(populated_db):
    di = _detected(populated_db)
    state = SearchState(query="x", collection_id="no-such-collection", results=[di])
    with pytest.raises(ValueError, match="no calibration"):
        _projector(populated_db).invoke(state)


def test_skips_results_without_mask(populated_db):
    # No mask → nothing to back-project → empty fused result.
    di = _detected(populated_db, masks_none=True)
    with pytest.warns(UserWarning, match="no mask"):
        out = _projector(populated_db).invoke(_state(populated_db, [di]))
    assert out.projected == []


def test_skips_when_no_depth_path(populated_db):
    di = _detected(populated_db).model_copy(update={"depth_path": ""})
    with pytest.warns(UserWarning, match="missing depth_path"):
        out = _projector(populated_db).invoke(_state(populated_db, [di]))
    assert out.projected == []


def test_raises_when_no_source_lists(populated_db):
    state = SearchState(query="x", collection_id=populated_db["collection_id"])
    with pytest.raises(ValueError, match="run the"):
        _projector(populated_db).invoke(state)


# ---------------------------------------------------------------------------
# cluster_instances mode
# ---------------------------------------------------------------------------


def _pose(xyz):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = xyz
    return m


def test_instances_mode_one_object_per_cluster(populated_db):
    # Two full-frame masks placed ~17 m apart → two spatial clusters.
    hits = [
        _detected(populated_db, frame=0),                            # near origin
        _detected(populated_db, frame=1, cam2world=_pose([10, 10, 10])),
    ]
    out = _projector(populated_db, mode="cluster_instances", min_instance_size=1).invoke(
        _state(populated_db, hits)
    )

    assert len(out.projected) == 2
    assert [o.id for o in out.projected] == ["obj-0", "obj-1"]
    for o in out.projected:
        assert o.points.shape[0] >= 1
        assert o.bbox is not None and o.bbox.shape == (2, 3)
    # The two clusters sit in clearly different world regions.
    c0 = out.projected[0].bbox.mean(axis=0)
    c1 = out.projected[1].bbox.mean(axis=0)
    assert np.linalg.norm(c0 - c1) > 5.0


def test_instances_sorted_largest_first(populated_db):
    # frame 0 → full 64-point cluster; frame 1 → 4-point cluster far away.
    small = torch.zeros(8, 8, dtype=torch.bool)
    small[0:2, 0:2] = True
    hits = [
        _detected(populated_db, frame=1, mask=small, cam2world=_pose([10, 10, 10])),
        _detected(populated_db, frame=0),   # deliberately last in input
    ]
    out = _projector(populated_db, mode="cluster_instances", min_instance_size=1).invoke(
        _state(populated_db, hits)
    )

    assert len(out.projected) == 2
    # Largest cluster first regardless of input order.
    assert out.projected[0].points.shape[0] == 64
    assert out.projected[1].points.shape[0] == 4


def test_instances_max_instances_caps_count(populated_db):
    hits = [
        _detected(populated_db, frame=0),
        _detected(populated_db, frame=1, cam2world=_pose([10, 10, 10])),
    ]
    out = _projector(
        populated_db, mode="cluster_instances", min_instance_size=1, max_instances=1
    ).invoke(_state(populated_db, hits))

    assert len(out.projected) == 1
    assert out.projected[0].id == "obj-0"


def test_instances_min_size_drops_small_cluster(populated_db):
    small = torch.zeros(8, 8, dtype=torch.bool)
    small[0:2, 0:2] = True   # 4 points
    hits = [
        _detected(populated_db, frame=0),                            # 64 points
        _detected(populated_db, frame=1, mask=small, cam2world=_pose([10, 10, 10])),
    ]
    out = _projector(
        populated_db, mode="cluster_instances", min_instance_size=10
    ).invoke(_state(populated_db, hits))

    assert len(out.projected) == 1
    assert out.projected[0].points.shape[0] == 64


def test_state_mode_overrides_constructor(populated_db):
    hits = [
        _detected(populated_db, frame=0),
        _detected(populated_db, frame=1, cam2world=_pose([10, 10, 10])),
    ]
    proj = _projector(populated_db, mode="cluster_single", min_instance_size=1)
    state = _state(populated_db, hits).model_copy(
        update={"mode": "cluster_instances"}
    )

    out = proj.invoke(state)

    assert len(out.projected) == 2


def test_single_mode_unchanged_by_default(populated_db):
    hits = [
        _detected(populated_db, frame=0),
        _detected(populated_db, frame=1, cam2world=_pose([10, 10, 10])),
    ]
    # Default mode="cluster_single" collapses everything to the largest cluster.
    out = _projector(populated_db).invoke(_state(populated_db, hits))
    assert len(out.projected) == 1
    assert out.projected[0].id == "fused"


def test_simple_mode_keeps_all_points_no_clustering(populated_db):
    # Two frames whose points are far apart (10 m). cluster_single would drop
    # one via DBSCAN; simple mode keeps every point in one object.
    hits = [
        _detected(populated_db, frame=0),
        _detected(populated_db, frame=1, cam2world=_pose([10, 10, 10])),
    ]
    # eps small enough that the two groups would NOT merge under DBSCAN.
    proj = ProjectTo3D(
        populated_db["db"], mode="simple", voxel=0.0
    )
    out = proj.invoke(_state(populated_db, hits))

    assert len(out.projected) == 1
    assert out.projected[0].id == "fused"
    assert out.projected[0].points.shape[0] == 128   # all points from both frames


def test_unknown_mode_raises(populated_db):
    with pytest.raises(ValueError, match="mode"):
        ProjectTo3D(populated_db["db"], mode="bogus")
