"""Tests for src.reconstruct.mock_vggt.

The mock's job is to honour the reconstruction contract — a FrameSet whose
frames, depth maps and poses are all usable downstream — not to reconstruct
anything. These tests pin that contract, so the real VGGT backend can be held
to the same one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.reconstruct import BaseReconstructor, MockVGGTReconstructor
from src.utils.datasets import FrameSet
from src.utils.geometry import build_scene_cloud


def test_returns_a_frameset_of_the_requested_length(scannet_tree):
    scene = scannet_tree(n=5)
    rec = MockVGGTReconstructor(scene_dir=scene)

    fs = rec.reconstruct(["a.jpg", "b.jpg", "c.jpg"], out_dir="unused")

    assert isinstance(fs, FrameSet)
    assert len(fs) == 3
    assert len(fs.depth_paths) == 3 and len(fs.poses) == 3


def test_output_is_usable_downstream(scannet_tree):
    """Every path exists, poses are finite 4x4, intrinsics + scale are present."""
    scene = scannet_tree(n=4)
    fs = MockVGGTReconstructor(scene_dir=scene).reconstruct(["x.jpg"] * 4, "unused")

    assert all(Path(p).is_file() for p in fs.paths)
    assert all(Path(p).is_file() for p in fs.depth_paths)
    for pose in fs.poses:
        assert pose.shape == (4, 4)
        assert np.isfinite(pose).all()
    assert set(fs.intrinsics) == {"fx", "fy", "cx", "cy"}
    assert fs.depth_scale > 0

    # Depth maps are 16-bit, as backproject expects.
    assert np.asarray(Image.open(fs.depth_paths[0])).dtype == np.uint16

    points, _ = build_scene_cloud(fs, n_frames=4, voxel=0.05)
    assert len(points) > 0


def test_frames_are_sampled_across_the_whole_scene(scannet_tree):
    """A short video must not collapse onto the first few frames of the scene."""
    scene = scannet_tree(n=10)
    fs = MockVGGTReconstructor(scene_dir=scene).reconstruct(["v.jpg"] * 3, "unused")

    picked = [Path(p).stem for p in fs.paths]
    assert picked == ["0", "4", "9"]


def test_requesting_more_frames_than_the_scene_has_is_clamped(scannet_tree):
    scene = scannet_tree(n=3)
    fs = MockVGGTReconstructor(scene_dir=scene).reconstruct(["v.jpg"] * 50, "unused")
    assert len(fs) == 3


def test_reports_progress_once_per_frame(scannet_tree):
    scene = scannet_tree(n=6)
    ticks: list[int] = []

    fs = MockVGGTReconstructor(scene_dir=scene).reconstruct(
        ["v.jpg"] * 4, "unused", on_progress=ticks.append
    )

    assert sum(ticks) == len(fs) == 4


def test_is_deterministic(scannet_tree):
    scene = scannet_tree(n=8)
    rec = MockVGGTReconstructor(scene_dir=scene)
    first = rec.reconstruct(["v.jpg"] * 5, "unused")
    second = rec.reconstruct(["v.jpg"] * 5, "unused")
    assert first.paths == second.paths


def test_writes_nothing_to_out_dir(scannet_tree, tmp_path):
    """It serves existing data — out_dir stays empty (real VGGT will write depth)."""
    scene = scannet_tree(n=3)
    out = tmp_path / "out"
    out.mkdir()

    MockVGGTReconstructor(scene_dir=scene).reconstruct(["v.jpg"] * 3, out)

    assert list(out.iterdir()) == []


def test_missing_scene_raises_with_an_actionable_message(tmp_path):
    rec = MockVGGTReconstructor(scene_dir=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="stands in for VGGT"):
        rec.reconstruct(["v.jpg"], "unused")


def test_implements_the_reconstructor_contract():
    assert issubclass(MockVGGTReconstructor, BaseReconstructor)
