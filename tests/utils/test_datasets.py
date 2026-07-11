"""Tests for src.utils.datasets (load_tum, load_scannet, load_collection)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from src.utils.datasets import FrameSet, load_collection, load_scannet, load_tum
from src.utils.db import connect


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


# ---------------------------------------------------------------------------
# load_collection (LanceDB → FrameSet)
# ---------------------------------------------------------------------------


def _make_collection(tmp_path, rows, *, with_meta=True, collection_id="coll"):
    """Create a LanceDB collection with the schema Indexer writes.

    Each entry of *rows* is ``(path, depth_path, cam2world_16)``.
    """
    db = connect(tmp_path / "lancedb")
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("collection_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 4)),
        pa.field("path", pa.string()),
        pa.field("depth_path", pa.string()),
        pa.field("cam2world", pa.list_(pa.float32())),
    ])
    table = db.create_table(collection_id, schema=schema)
    table.add([
        {
            "id": f"id-{i}",
            "collection_id": collection_id,
            "vector": [0.0, 0.0, 0.0, 1.0],
            "path": path,
            "depth_path": depth_path,
            "cam2world": cam,
        }
        for i, (path, depth_path, cam) in enumerate(rows)
    ])

    if with_meta:
        meta_schema = pa.schema([
            pa.field("collection_id", pa.string()),
            pa.field("fx", pa.float64()),
            pa.field("fy", pa.float64()),
            pa.field("cx", pa.float64()),
            pa.field("cy", pa.float64()),
            pa.field("depth_scale", pa.float64()),
        ])
        meta = db.create_table("_collection_meta", schema=meta_schema)
        meta.add([{
            "collection_id": collection_id,
            "fx": 500.0, "fy": 510.0, "cx": 320.0, "cy": 240.0,
            "depth_scale": 1000.0,
        }])
    return db


IDENTITY16 = np.eye(4, dtype=np.float32).reshape(16).tolist()


def test_load_collection_rebuilds_the_frameset(tmp_path):
    db = _make_collection(tmp_path, [
        (f"color/{i}.jpg", f"depth/{i}.png", IDENTITY16) for i in range(3)
    ])
    fs = load_collection(db, "coll")

    assert isinstance(fs, FrameSet)
    assert len(fs) == 3
    assert fs.depth_paths == [f"depth/{i}.png" for i in range(3)]
    assert fs.intrinsics == {"fx": 500.0, "fy": 510.0, "cx": 320.0, "cy": 240.0}
    assert fs.depth_scale == 1000.0
    for pose in fs.poses:
        assert pose.shape == (4, 4)
        np.testing.assert_allclose(pose, np.eye(4))


def test_load_collection_orders_frames_naturally(tmp_path):
    """frame2 sorts before frame10 — lexicographic order would invert them."""
    db = _make_collection(tmp_path, [
        (f"color/{i}.jpg", f"depth/{i}.png", IDENTITY16) for i in (10, 2, 1)
    ])
    fs = load_collection(db, "coll")
    assert fs.paths == ["color/1.jpg", "color/2.jpg", "color/10.jpg"]


def test_load_collection_preserves_pose_values(tmp_path):
    pose = np.arange(16, dtype=np.float32).reshape(4, 4)
    pose[3] = [0, 0, 0, 1]
    db = _make_collection(tmp_path, [("c/0.jpg", "d/0.png", pose.reshape(16).tolist())])
    fs = load_collection(db, "coll")
    np.testing.assert_allclose(fs.poses[0], pose)


def test_load_collection_skips_unusable_rows(tmp_path):
    """Non-finite poses and rows missing a depth path are dropped."""
    bad_pose = np.full(16, np.inf, dtype=np.float32).tolist()
    db = _make_collection(tmp_path, [
        ("color/0.jpg", "depth/0.png", IDENTITY16),
        ("color/1.jpg", "depth/1.png", bad_pose),
        ("color/2.jpg", "", IDENTITY16),
    ])
    fs = load_collection(db, "coll")
    assert fs.paths == ["color/0.jpg"]


def test_load_collection_without_meta_raises(tmp_path):
    db = _make_collection(tmp_path, [("c/0.jpg", "d/0.png", IDENTITY16)], with_meta=False)
    with pytest.raises(ValueError, match="calibration"):
        load_collection(db, "coll")


def test_load_collection_unknown_collection_raises(tmp_path):
    db = _make_collection(tmp_path, [("c/0.jpg", "d/0.png", IDENTITY16)])
    with pytest.raises(ValueError, match="not in this store"):
        load_collection(db, "nope")
