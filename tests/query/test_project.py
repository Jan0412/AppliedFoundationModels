"""Tests for src.query.project.ProjectTo3D (multi-frame fusion → one box)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data_model import DetectedImage, SearchState
from src.query.project import ProjectTo3D


def _detected(populated_db, *, frame=0, mask=None, masks_none=False, boxes=None):
    """Build a DetectedImage wired to one frame of the populated_db fixture."""
    if mask is None:
        mask = torch.ones(8, 8, dtype=torch.bool)
    if boxes is None:
        boxes = torch.tensor([[0.0, 0.0, 8.0, 8.0]])
    return DetectedImage(
        id=populated_db["ids"][frame],
        path=populated_db["paths"][frame],
        similarity_score=1.0,
        detection_score=0.9,
        boxes=boxes,
        scores=torch.tensor([0.9] * len(boxes)),
        masks=None if masks_none else [mask],
        depth_path=populated_db["depth_paths"][frame],
        cam2world=np.eye(4, dtype=np.float32),
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


def test_projects_box_only_detection(populated_db):
    # masks=None but a full-frame box → box path rasterises it and projects.
    di = _detected(populated_db, masks_none=True)
    out = _projector(populated_db).invoke(_state(populated_db, [di]))
    obj = out.projected[0]
    assert obj.points.shape == (64, 3)        # full-frame box → all pixels
    assert obj.bbox is not None and obj.bbox.shape == (2, 3)


def test_box_only_smaller_box_projects_fewer_points(populated_db):
    box = torch.tensor([[1.0, 2.0, 4.0, 5.0]])   # width 3 * height 3 = 9 px
    di = _detected(populated_db, masks_none=True, boxes=box)
    out = _projector(populated_db).invoke(_state(populated_db, [di]))
    assert out.projected[0].points.shape == (9, 3)


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


def test_skips_results_without_masks_or_boxes(populated_db):
    # Neither masks nor boxes → nothing to back-project → empty fused result.
    di = _detected(populated_db, masks_none=True, boxes=torch.zeros(0, 4))
    with pytest.warns(UserWarning, match="no masks or boxes"):
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
