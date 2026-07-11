"""Tests for src.ui.render pure helpers (no viser server needed)."""

from __future__ import annotations

import numpy as np
import torch

from src.data_model import (
    DetectedImage,
    ProjectedObject,
    RetrievalDiagnostics,
    SearchState,
)
from src.ui import render


def _state(**kw):
    base = dict(query="chair", collection_id="room")
    base.update(kw)
    return SearchState(**base)


def _hit(detection_score: float):
    """A minimal DetectedImage carrying a SAM detection score."""
    return DetectedImage(
        id="x", path="p.png", similarity_score=0.1,
        detection_score=detection_score,
        boxes=torch.zeros((0, 4)), scores=torch.zeros((0,)),
    )


def _obj():
    return ProjectedObject(
        id="obj-0", path="",
        points=np.zeros((5, 3), dtype=np.float32), colors=None,
        bbox=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
    )


def test_object_color_cycles_through_the_palette():
    assert render.object_color(0) == render.OBJ_COLORS[0]
    assert render.object_color(len(render.OBJ_COLORS)) == render.OBJ_COLORS[0]


def test_box_edges_reference_all_eight_corners():
    assert render.BOX_EDGES.shape == (12, 2)
    assert set(render.BOX_EDGES.flatten()) == set(range(8))


def test_results_markdown_reports_objects_and_diagnostics():
    diag = RetrievalDiagnostics(
        mode="dynamic", pool_size=12, total_frames=200,
        n_selected=10, separability=0.83,
    )
    md = render.results_markdown(_state(
        projected=[_obj()],
        results=[_hit(0.87)],
        retrieval_diag=diag,
    ))
    assert "1 object" in md
    assert "chair" in md
    assert "12/200" in md
    assert "dynamic" in md
    assert "detection 0.87" in md      # SAM confidence surfaced
    assert "η 0.83" in md              # separability kept as a diagnostic


def test_best_detection_score_is_the_max_over_results():
    state = _state(results=[_hit(0.4), _hit(0.9), _hit(0.6)])
    assert render.best_detection_score(state) == 0.9
    assert render.best_detection_score(_state(results=[])) == 0.0


def test_results_markdown_handles_no_result():
    md = render.results_markdown(_state(projected=[], results=[]))
    assert "No 3D object" in md


def test_warns_when_sam_detects_nothing():
    """best == 0 → the object was grounded in no frame → probably absent."""
    md = render.results_markdown(_state(projected=[], results=[_hit(0.0)]))
    assert "did not detect" in md.lower()
    assert "separability" not in md.lower()   # no longer the SigLIP-based warning


def test_warns_on_weak_detection():
    md = render.results_markdown(
        _state(projected=[_obj()], results=[_hit(0.53)]),
        detection_warn_threshold=0.6,
    )
    assert "weak detection" in md.lower()
    assert "0.53" in md


def test_confident_detection_has_no_warning():
    md = render.results_markdown(
        _state(projected=[_obj()], results=[_hit(0.92)]),
        detection_warn_threshold=0.6,
    )
    assert "⚠️" not in md


def test_detected_in_2d_but_not_placed_in_3d():
    """SAM grounded it (best > 0) but nothing back-projected (no object)."""
    md = render.results_markdown(_state(projected=[], results=[_hit(0.8)]))
    assert "2D" in md
    assert "did not detect" not in md.lower()   # it WAS detected, just not placed


def test_mask_overlay_draws_on_the_frame(tmp_path):
    from PIL import Image
    frame = tmp_path / "f.png"
    Image.new("RGB", (64, 48), color=(0, 0, 0)).save(frame)

    mask = torch.zeros((48, 64), dtype=torch.bool)
    mask[10:30, 10:30] = True
    hit = DetectedImage(
        id="x", path=str(frame), similarity_score=0.9, detection_score=0.8,
        boxes=torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
        scores=torch.tensor([0.8]),
        masks=[mask],
    )

    out = render.mask_overlay(hit, "chair", width=64)
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8
    # The overlay painted colour onto a previously black frame.
    assert out.sum() > 0


def test_mask_overlay_scales_down_wide_frames(tmp_path):
    from PIL import Image
    frame = tmp_path / "big.png"
    Image.new("RGB", (640, 480), color=(20, 20, 20)).save(frame)
    hit = DetectedImage(
        id="x", path=str(frame), similarity_score=0.5, detection_score=0.5,
        boxes=torch.zeros((0, 4)), scores=torch.zeros((0,)), masks=[],
    )
    out = render.mask_overlay(hit, "q", width=320)
    assert out.shape[1] == 320
    assert out.shape[0] == 240        # aspect ratio preserved


def test_evidence_caption_mentions_both_scores():
    hit = DetectedImage(
        id="x", path="p.png", similarity_score=0.912, detection_score=0.734,
        boxes=torch.zeros((0, 4)), scores=torch.zeros((0,)),
    )
    caption = render.evidence_caption(2, hit)
    assert "#2" in caption
    assert "0.912" in caption
    assert "0.734" in caption
