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


def test_object_color_cycles_through_the_palette():
    assert render.object_color(0) == render.OBJ_COLORS[0]
    assert render.object_color(len(render.OBJ_COLORS)) == render.OBJ_COLORS[0]


def test_box_edges_reference_all_eight_corners():
    assert render.BOX_EDGES.shape == (12, 2)
    assert set(render.BOX_EDGES.flatten()) == set(range(8))


def test_results_markdown_reports_objects_and_diagnostics():
    obj = ProjectedObject(
        id="obj-0",
        path="",
        points=np.zeros((5, 3), dtype=np.float32),
        colors=None,
        bbox=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
    )
    diag = RetrievalDiagnostics(
        mode="dynamic", pool_size=12, total_frames=200,
        n_selected=10, separability=0.83,
    )
    md = render.results_markdown(_state(
        projected=[obj],
        results=[],
        retrieval_diag=diag,
    ))
    assert "1 object" in md
    assert "chair" in md
    assert "12/200" in md
    assert "dynamic" in md


def test_results_markdown_handles_no_result():
    md = render.results_markdown(_state(projected=[], results=[]))
    assert "No 3D object" in md


def test_results_markdown_warns_when_gated():
    diag = RetrievalDiagnostics(mode="dynamic", pool_size=10, gated=True)
    md = render.results_markdown(_state(projected=[], results=[], retrieval_diag=diag))
    assert "separability" in md.lower()


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
