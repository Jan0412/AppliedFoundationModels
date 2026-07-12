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


# ---------------------------------------------------------------------------
# Scene / highlight nodes (against the fake viser server)
# ---------------------------------------------------------------------------


def test_show_scene_draws_under_the_stable_node_name(fake_server):
    """One fixed name: re-adding replaces the cloud instead of stacking clouds."""
    points = np.zeros((4, 3), dtype=np.float32)

    render.show_scene(fake_server, points, None, point_size=0.02)
    render.show_scene(fake_server, points, None, point_size=0.02)

    assert fake_server.scene.names_of("point_cloud") == ["/scene", "/scene"]
    node = fake_server.scene.nodes[0]
    assert node.colors == (160, 160, 160)      # grey stand-in when uncoloured
    assert node.point_size == 0.02


def test_show_scene_passes_colors_through_when_present(fake_server):
    colors = np.full((4, 3), 7, dtype=np.uint8)
    render.show_scene(
        fake_server, np.zeros((4, 3), dtype=np.float32), colors, point_size=0.01
    )
    np.testing.assert_array_equal(fake_server.scene.nodes[0].colors, colors)


def test_show_highlights_draws_cloud_box_and_label_per_object(fake_server):
    handles = render.show_highlights(
        fake_server, [_obj(), _obj_with("obj-1", 3)], point_size=0.02, query="chair"
    )

    assert len(handles) == 6            # cloud + box + label, twice
    assert fake_server.scene.names_of("point_cloud") == [
        "/highlight/obj-0", "/highlight/obj-1"
    ]
    assert fake_server.scene.names_of("line_segments") == [
        "/highlight/box-obj-0", "/highlight/box-obj-1"
    ]
    # Highlights sit above the scene cloud, so they get a fatter point.
    assert fake_server.scene.nodes[0].point_size == 0.02 * 1.5
    # 12 edges × 2 endpoints × xyz.
    assert fake_server.scene.nodes[1].points.shape == (12, 2, 3)

    labels = [n for n in fake_server.scene.nodes if n.kind == "label"]
    assert labels[0].text == "chair (obj-0)"
    # The label floats at the top of the box, not at its centre.
    assert labels[0].position[2] == 1.0


def test_highlight_label_falls_back_to_the_object_id(fake_server):
    render.show_highlights(fake_server, [_obj()], point_size=0.02, query="")
    label = [n for n in fake_server.scene.nodes if n.kind == "label"][0]
    assert label.text == "obj-0"


def test_object_without_a_bbox_gets_points_but_no_box(fake_server):
    """Projection can yield a cloud with no usable box — draw what there is."""
    obj = ProjectedObject(
        id="obj-0", path="",
        points=np.zeros((5, 3), dtype=np.float32), colors=None, bbox=None,
    )
    handles = render.show_highlights(fake_server, [obj], point_size=0.02)

    assert len(handles) == 1
    assert fake_server.scene.names_of("line_segments") == []
    assert fake_server.scene.names_of("label") == []


def test_clear_highlights_removes_every_handle(fake_server):
    handles = render.show_highlights(fake_server, [_obj()], point_size=0.02)
    render.clear_highlights(handles)
    assert all(h.removed for h in handles)


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


def test_mask_overlay_falls_back_to_the_default_font(tmp_path, monkeypatch):
    """DejaVu is a Debian path — elsewhere the boxes must still get their labels."""
    from PIL import Image, ImageFont

    real_truetype = ImageFont.truetype

    def _no_font_file(font=None, *args, **kwargs):
        # Only the on-disk DejaVu lookup fails; load_default()'s bundled font
        # (a BytesIO) still resolves, which is the fallback under test.
        if isinstance(font, str):
            raise OSError("cannot open resource")
        return real_truetype(font, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", _no_font_file)

    frame = tmp_path / "f.png"
    Image.new("RGB", (64, 48), color=(0, 0, 0)).save(frame)
    hit = DetectedImage(
        id="x", path=str(frame), similarity_score=0.9, detection_score=0.8,
        boxes=torch.tensor([[5.0, 5.0, 40.0, 40.0]]),
        scores=torch.tensor([0.8]),
        masks=[],
    )

    out = render.mask_overlay(hit, "chair", width=64)

    assert out.shape == (48, 64, 3)
    assert out.sum() > 0        # the box was still drawn


def test_evidence_caption_mentions_both_scores():
    hit = DetectedImage(
        id="x", path="p.png", similarity_score=0.912, detection_score=0.734,
        boxes=torch.zeros((0, 4)), scores=torch.zeros((0,)),
    )
    caption = render.evidence_caption(2, hit)
    assert "#2" in caption
    assert "0.912" in caption
    assert "0.734" in caption


# ---------------------------------------------------------------------------
# objects_table
# ---------------------------------------------------------------------------


def _obj_with(obj_id: str, n_points: int):
    return ProjectedObject(
        id=obj_id, path="",
        points=np.zeros((n_points, 3), dtype=np.float32), colors=None,
        bbox=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
    )


def test_objects_table_lists_point_counts_in_order():
    table = render.objects_table([_obj_with("obj-0", 12847), _obj_with("obj-1", 3201)])
    lines = table.splitlines()

    assert lines[0].startswith("| Object | Points |")
    assert "| obj-0 | 12,847 |" in lines        # thousands separated
    assert "| obj-1 | 3,201 |" in lines
    # Input order (largest-first) is preserved, matching the highlight colours.
    assert lines.index("| obj-0 | 12,847 |") < lines.index("| obj-1 | 3,201 |")


def test_objects_table_is_empty_when_nothing_projected():
    assert render.objects_table([]) == ""


def test_results_markdown_appends_the_objects_table():
    md = render.results_markdown(_state(
        projected=[_obj_with("obj-0", 500)], results=[_hit(0.9)],
    ))
    assert "| Object | Points |" in md
    assert "| obj-0 | 500 |" in md


def test_results_markdown_has_no_table_without_objects():
    md = render.results_markdown(_state(projected=[], results=[_hit(0.9)]))
    assert "| Object | Points |" not in md
