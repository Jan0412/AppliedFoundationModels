"""Tests for src.utils.datasets (load_tum, load_scannet)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.utils.datasets import FrameSet, load_scannet, load_tum


# ---------------------------------------------------------------------------
# load_tum
# ---------------------------------------------------------------------------


def _make_tum_tree(root, n=3, dt_depth=0.001, dt_gt=0.001):
    """Write a minimal TUM sequence with n index-aligned frames."""
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    rgb_lines, depth_lines, gt_lines = [], [], []
    for i in range(n):
        ts = 100.0 + i
        Image.new("RGB", (4, 4), color=(i, 0, 0)).save(root / "rgb" / f"{i}.png")
        Image.fromarray(np.zeros((4, 4), np.uint16)).save(root / "depth" / f"{i}.png")
        rgb_lines.append(f"{ts:.4f} rgb/{i}.png")
        depth_lines.append(f"{ts + dt_depth:.4f} depth/{i}.png")
        # identity quaternion (qw=1), translation = (i, 0, 0)
        gt_lines.append(f"{ts + dt_gt:.4f} {i} 0 0 0 0 0 1")
    (root / "rgb.txt").write_text("# comment\n" + "\n".join(rgb_lines) + "\n")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n")
    (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n")


def test_load_tum_returns_frameset_aligned(tmp_path):
    _make_tum_tree(tmp_path, n=3)
    fs = load_tum(tmp_path, intrinsics={"fx": 1, "fy": 2, "cx": 3, "cy": 4})

    assert isinstance(fs, FrameSet)
    assert len(fs) == 3
    assert len(fs.depth_paths) == 3 and len(fs.poses) == 3
    assert all(p.endswith(f"rgb/{i}.png") for i, p in enumerate(fs.paths))
    assert all(p.endswith(f"depth/{i}.png") for i, p in enumerate(fs.depth_paths))


def test_load_tum_translations_from_groundtruth(tmp_path):
    _make_tum_tree(tmp_path, n=3)
    fs = load_tum(tmp_path, intrinsics={"fx": 1, "fy": 1, "cx": 0, "cy": 0})
    for i, pose in enumerate(fs.poses):
        assert pose.shape == (4, 4)
        assert np.allclose(pose[:3, 3], [i, 0, 0])


def test_load_tum_passes_intrinsics_and_default_scale(tmp_path):
    _make_tum_tree(tmp_path, n=2)
    intr = {"fx": 5.0, "fy": 6.0, "cx": 7.0, "cy": 8.0}
    fs = load_tum(tmp_path, intrinsics=intr)
    assert fs.intrinsics == intr
    assert fs.depth_scale == 5000.0


def test_load_tum_drops_unassociated_frames(tmp_path):
    # Make the depth timestamp for one frame fall far outside max_dt.
    _make_tum_tree(tmp_path, n=3)
    lines = (tmp_path / "depth.txt").read_text().splitlines()
    parts = lines[1].split()
    parts[0] = "999999.0"   # frame 1 depth is now unmatchable
    lines[1] = " ".join(parts)
    (tmp_path / "depth.txt").write_text("\n".join(lines) + "\n")

    fs = load_tum(tmp_path, intrinsics={"fx": 1, "fy": 1, "cx": 0, "cy": 0})
    assert len(fs) == 2   # frame 1 dropped


# ---------------------------------------------------------------------------
# load_scannet
# ---------------------------------------------------------------------------


def _make_scannet_tree(root, n=3, bad_pose_index=None):
    (root / "color").mkdir(parents=True)
    (root / "depth").mkdir(parents=True)
    (root / "pose").mkdir(parents=True)
    (root / "intrinsic").mkdir(parents=True)
    K = np.eye(4)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = 500.0, 510.0, 320.0, 240.0
    np.savetxt(root / "intrinsic" / "intrinsic_depth.txt", K)
    for i in range(n):
        Image.new("RGB", (4, 4), color=(i, 0, 0)).save(root / "color" / f"{i}.jpg")
        Image.fromarray(np.zeros((4, 4), np.uint16)).save(root / "depth" / f"{i}.png")
        pose = np.eye(4)
        pose[:3, 3] = [i, 0, 0]
        if bad_pose_index is not None and i == bad_pose_index:
            pose[:] = -np.inf
        np.savetxt(root / "pose" / f"{i}.txt", pose)


def test_load_scannet_returns_frameset(tmp_path):
    _make_scannet_tree(tmp_path, n=3)
    fs = load_scannet(tmp_path)
    assert len(fs) == 3
    assert fs.intrinsics == {"fx": 500.0, "fy": 510.0, "cx": 320.0, "cy": 240.0}
    assert fs.depth_scale == 1000.0
    assert all(p.endswith(f"color/{i}.jpg") for i, p in enumerate(fs.paths))


def test_load_scannet_skips_nonfinite_poses(tmp_path):
    _make_scannet_tree(tmp_path, n=3, bad_pose_index=1)
    fs = load_scannet(tmp_path)
    assert len(fs) == 2   # frame 1's inf pose dropped


def test_load_scannet_poses_are_4x4(tmp_path):
    _make_scannet_tree(tmp_path, n=2)
    fs = load_scannet(tmp_path)
    for pose in fs.poses:
        assert pose.shape == (4, 4)
