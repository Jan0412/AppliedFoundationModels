"""Tests for src.utils.geometry (quat_to_cam2world, backproject)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.utils.geometry import (
    aabb_corners,
    backproject,
    build_scene_cloud,
    central_depth,
    cluster_masks,
    farthest_point_sampling,
    largest_cluster,
    lookat_point,
    quat_to_cam2world,
    robust_bounds,
    voxel_downsample,
)


# ---------------------------------------------------------------------------
# quat_to_cam2world
# ---------------------------------------------------------------------------


def test_identity_quaternion_gives_identity_matrix():
    M = quat_to_cam2world(0, 0, 0, 0, 0, 0, 1)
    assert np.allclose(M, np.eye(4))


def test_translation_is_placed_in_last_column():
    M = quat_to_cam2world(1.0, 2.0, 3.0, 0, 0, 0, 1)
    assert np.allclose(M[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(M[:3, :3], np.eye(3))


def test_rotation_is_orthonormal():
    # 90° about Z: qz = sin(45°), qw = cos(45°)
    s = np.sqrt(0.5)
    R = quat_to_cam2world(0, 0, 0, 0, 0, s, s)[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# backproject
# ---------------------------------------------------------------------------


def _write_depth(path, arr_u16):
    Image.fromarray(np.asarray(arr_u16, dtype=np.uint16)).save(path)


def test_backproject_known_xyz_identity_pose(tmp_path):
    # 3x3 depth, single masked pixel at (y=1, x=2). With fx=fy=1, cx=cy=0,
    # depth_scale=1 and identity pose: X=u*Z, Y=v*Z, Z=Z.
    depth = np.full((3, 3), 2, dtype=np.uint16)
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)

    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 2] = True

    pts, colors = backproject(
        mask, dpath, np.eye(4), intr=(1.0, 1.0, 0.0, 0.0), depth_scale=1.0
    )
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0], [4.0, 2.0, 2.0])   # (2*2, 1*2, 2)
    assert colors is None   # no rgb_path supplied


def test_backproject_applies_world_translation(tmp_path):
    depth = np.full((2, 2), 1, dtype=np.uint16)
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True   # pixel (u=0, v=0) → camera point (0, 0, 1)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    pts, _ = backproject(mask, dpath, pose, intr=(1.0, 1.0, 0.0, 0.0), depth_scale=1.0)
    assert np.allclose(pts[0], [10.0, 20.0, 31.0])


def test_backproject_skips_zero_depth(tmp_path):
    depth = np.array([[0, 5], [0, 0]], dtype=np.uint16)
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)
    mask = np.ones((2, 2), dtype=bool)
    pts, _ = backproject(mask, dpath, np.eye(4), intr=(1.0, 1.0, 0.0, 0.0), depth_scale=1.0)
    assert pts.shape == (1, 3)   # only the single non-zero-depth pixel survives


def test_backproject_samples_colors_when_rgb_given(tmp_path):
    depth = np.full((2, 2), 1, dtype=np.uint16)
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 1] = [10, 20, 30]
    rpath = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(rpath)

    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 1] = True
    pts, colors = backproject(
        mask, dpath, np.eye(4), intr=(1.0, 1.0, 0.0, 0.0), depth_scale=1.0,
        rgb_path=rpath,
    )
    assert colors is not None
    assert colors.shape == (1, 3)
    assert list(colors[0]) == [10, 20, 30]


def test_backproject_empty_mask_returns_empty(tmp_path):
    depth = np.full((2, 2), 1, dtype=np.uint16)
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)
    mask = np.zeros((2, 2), dtype=bool)
    pts, _ = backproject(mask, dpath, np.eye(4), intr=(1.0, 1.0, 0.0, 0.0), depth_scale=1.0)
    assert pts.shape == (0, 3)


def test_backproject_resizes_mask_to_depth_resolution(tmp_path):
    # depth 2x2; mask 4x4 (2x upscale). After resize, all depth pixels masked.
    depth = np.full((2, 2), 1, dtype=np.uint16)
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)
    mask = np.ones((4, 4), dtype=bool)   # larger than depth
    pts, _ = backproject(mask, dpath, np.eye(4), intr=(1.0, 1.0, 0.0, 0.0), depth_scale=1.0)
    assert pts.shape == (4, 3)           # 2x2 depth pixels all valid


def test_backproject_scales_color_indices_when_resolutions_differ(tmp_path):
    # depth 2x2; rgb 4x4.  Mask at depth size, identity pose.
    depth = np.full((2, 2), 1, dtype=np.uint16)
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    # Colour pixel at (row=0, col=2) in RGB space should map to depth (row=0, col=1)
    rgb[0, 2] = [99, 88, 77]
    rpath = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(rpath)
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 1] = True   # depth pixel (row=0, col=1)
    pts, colors = backproject(mask, dpath, np.eye(4), intr=(1.0, 1.0, 0.0, 0.0),
                              depth_scale=1.0, rgb_path=rpath)
    assert pts.shape == (1, 3)
    assert colors is not None
    # col=1 in depth (W=2) → col=2 in RGB (W=4): factor 4/2=2
    assert list(colors[0]) == [99, 88, 77]


# ---------------------------------------------------------------------------
# robust_bounds
# ---------------------------------------------------------------------------


def test_robust_bounds_returns_min_max():
    pts = np.array([[0, 0, 0], [1, 2, 3], [-1, -2, -3]], dtype=np.float32)
    b = robust_bounds(pts, lo=0, hi=100)
    assert b.shape == (2, 3)
    assert np.allclose(b[0], [-1, -2, -3])
    assert np.allclose(b[1], [1, 2, 3])


def test_robust_bounds_trims_outlier():
    # 100 points in [0,1] plus one far outlier; 2–98 pct excludes the outlier.
    base = np.tile(np.linspace(0, 1, 100)[:, None], (1, 3)).astype(np.float32)
    pts = np.vstack([base, [[1000, 1000, 1000]]]).astype(np.float32)
    trimmed = robust_bounds(pts, lo=2, hi=98)
    raw = robust_bounds(pts, lo=0, hi=100)
    assert trimmed[1, 0] < raw[1, 0]                  # upper bound pulled in
    assert trimmed[1, 0] < 2.0                        # outlier excluded


def test_robust_bounds_empty_returns_none():
    assert robust_bounds(np.empty((0, 3), dtype=np.float32)) is None


# ---------------------------------------------------------------------------
# aabb_corners
# ---------------------------------------------------------------------------


def test_aabb_corners_shape_and_extremes():
    bbox = np.array([[0, 0, 0], [2, 4, 6]], dtype=np.float32)
    corners = aabb_corners(bbox)
    assert corners.shape == (8, 3)
    assert np.allclose(corners.min(axis=0), [0, 0, 0])
    assert np.allclose(corners.max(axis=0), [2, 4, 6])
    # all 8 unique corners present
    assert len({tuple(c) for c in corners.tolist()}) == 8


# ---------------------------------------------------------------------------
# voxel_downsample
# ---------------------------------------------------------------------------


def test_voxel_downsample_dedups_coincident_points():
    pts = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    out, _ = voxel_downsample(pts, voxel=0.1)
    assert len(out) == 2                       # 3 coincident → 1, plus the other


def test_voxel_downsample_keeps_colors_aligned():
    pts = np.array([[0, 0, 0], [5, 5, 5]], dtype=np.float32)
    cols = np.array([[10, 0, 0], [0, 20, 0]], dtype=np.uint8)
    out_p, out_c = voxel_downsample(pts, cols, voxel=0.1)
    assert len(out_p) == len(out_c) == 2


def test_voxel_downsample_disabled_when_voxel_nonpositive():
    pts = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
    out, _ = voxel_downsample(pts, voxel=0.0)
    assert len(out) == 2                       # no dedup


# ---------------------------------------------------------------------------
# largest_cluster
# ---------------------------------------------------------------------------


def test_largest_cluster_selects_dense_blob_drops_outliers():
    rng = np.random.default_rng(0)
    blob = rng.normal(scale=0.01, size=(60, 3)).astype(np.float32)      # tight
    outliers = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=np.float32)
    pts = np.vstack([blob, outliers])
    mask = largest_cluster(pts, eps=0.05, min_samples=5)
    assert mask[:60].all()                     # blob kept
    assert not mask[60:].any()                 # far outliers dropped


def test_largest_cluster_fallback_when_all_noise():
    pts = np.array([[0, 0, 0], [9, 9, 9], [18, 18, 18]], dtype=np.float32)
    mask = largest_cluster(pts, eps=0.05, min_samples=5)
    assert mask.all()                          # no cluster → keep everything


def test_largest_cluster_empty_returns_empty():
    mask = largest_cluster(np.empty((0, 3), dtype=np.float32))
    assert mask.shape == (0,)


# ---------------------------------------------------------------------------
# build_scene_cloud
# ---------------------------------------------------------------------------


def _make_frameset(tmp_path, n=3):
    """Minimal FrameSet-like object: n frames, 4x4 depth at 1 m, identity poses."""
    from dataclasses import dataclass

    @dataclass
    class _FS:
        paths: list
        depth_paths: list
        poses: list
        intrinsics: dict
        depth_scale: float

    img_dir = tmp_path / "rgb"
    dep_dir = tmp_path / "dep"
    img_dir.mkdir(parents=True)
    dep_dir.mkdir(parents=True)

    paths, depth_paths, poses = [], [], []
    for i in range(n):
        rgb_p = img_dir / f"{i}.png"
        dep_p = dep_dir / f"{i}.png"
        Image.new("RGB", (4, 4), color=(i * 50, 0, 0)).save(rgb_p)
        Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16)).save(dep_p)
        paths.append(str(rgb_p))
        depth_paths.append(str(dep_p))
        poses.append(np.eye(4, dtype=np.float32))

    return _FS(paths, depth_paths, poses,
               {"fx": 2.0, "fy": 2.0, "cx": 2.0, "cy": 2.0}, 1000.0)


def test_build_scene_cloud_returns_nonempty(tmp_path):
    fs = _make_frameset(tmp_path, n=3)
    pts, cols = build_scene_cloud(fs, n_frames=3, voxel=0.0)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert len(pts) > 0
    assert cols is not None and cols.shape == pts.shape


def test_build_scene_cloud_voxel_reduces_points(tmp_path):
    fs = _make_frameset(tmp_path, n=3)
    pts_fine, _ = build_scene_cloud(fs, n_frames=3, voxel=0.0)
    pts_coarse, _ = build_scene_cloud(fs, n_frames=3, voxel=1.0)
    assert len(pts_coarse) <= len(pts_fine)


def test_build_scene_cloud_empty_frameset():
    from dataclasses import dataclass

    @dataclass
    class _FS:
        paths: list = ()
        depth_paths: list = ()
        poses: list = ()
        intrinsics: dict = None
        depth_scale: float = 1000.0

    pts, cols = build_scene_cloud(_FS(), n_frames=10, voxel=0.02)
    assert pts.shape == (0, 3)
    assert cols is None


# ---------------------------------------------------------------------------
# central_depth
# ---------------------------------------------------------------------------


def test_central_depth_median_of_central_patch(tmp_path):
    # 10x10 with patch_frac=0.2 → 2x2 patch at [4:6, 4:6].
    depth = np.zeros((10, 10), dtype=np.uint16)
    depth[4:6, 4:6] = [[1000, 2000], [3000, 4000]]
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)

    assert central_depth(dpath, depth_scale=1000.0, patch_frac=0.2) == 2.5


def test_central_depth_excludes_zero_pixels(tmp_path):
    depth = np.zeros((10, 10), dtype=np.uint16)
    depth[4:6, 4:6] = [[0, 0], [2000, 4000]]
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)

    assert central_depth(dpath, depth_scale=1000.0, patch_frac=0.2) == 3.0


def test_central_depth_ignores_pixels_outside_patch(tmp_path):
    # Everything outside the central 2x2 screams 60 m; the patch says 1 m.
    depth = np.full((10, 10), 60_000, dtype=np.uint16)
    depth[4:6, 4:6] = 1000
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)

    assert central_depth(dpath, depth_scale=1000.0, patch_frac=0.2) == 1.0


def test_central_depth_none_when_patch_has_no_valid_depth(tmp_path):
    depth = np.full((10, 10), 3000, dtype=np.uint16)
    depth[4:6, 4:6] = 0
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)

    assert central_depth(dpath, depth_scale=1000.0, patch_frac=0.2) is None


def test_central_depth_patch_is_at_least_one_pixel(tmp_path):
    # 2x2 image with patch_frac=0.2 → 1x1 patch at [0:1, 0:1].
    depth = np.array([[1500, 0], [0, 0]], dtype=np.uint16)
    dpath = tmp_path / "d.png"
    _write_depth(dpath, depth)

    assert central_depth(dpath, depth_scale=1000.0, patch_frac=0.2) == 1.5


# ---------------------------------------------------------------------------
# lookat_point
# ---------------------------------------------------------------------------


def test_lookat_point_identity_pose_is_along_z():
    assert np.allclose(lookat_point(np.eye(4), 2.5), [0.0, 0.0, 2.5])


def test_lookat_point_follows_rotated_axis_and_translation():
    # Optical axis (+Z column) remapped to world +X, camera at (1, 2, 3).
    M = np.eye(4, dtype=np.float32)
    M[:3, 0] = [0, 1, 0]
    M[:3, 1] = [0, 0, 1]
    M[:3, 2] = [1, 0, 0]
    M[:3, 3] = [1, 2, 3]

    assert np.allclose(lookat_point(M, 2.0), [3.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# farthest_point_sampling
# ---------------------------------------------------------------------------


def _circle_dirs(n: int) -> np.ndarray:
    angles = 2.0 * np.pi * np.arange(n) / n
    return np.stack(
        [np.cos(angles), np.sin(angles), np.zeros(n)], axis=1
    ).astype(np.float32)


def test_fps_picks_evenly_spread_directions():
    # 8 directions every 45°; n=4 from seed 0 → exactly 0°, 90°, 180°, 270°.
    picked = farthest_point_sampling(_circle_dirs(8), seed_idx=0, n=4)
    assert picked[0] == 0
    assert set(picked) == {0, 2, 4, 6}


def test_fps_returns_all_when_n_exceeds_population():
    picked = farthest_point_sampling(_circle_dirs(3), seed_idx=1, n=10)
    assert picked[0] == 1
    assert sorted(picked) == [0, 1, 2]


def test_fps_prefers_distinct_over_duplicate_directions():
    dirs = np.array(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
    )
    picked = farthest_point_sampling(dirs, seed_idx=0, n=2)
    assert picked == [0, 2]


def test_fps_never_repeats_an_index():
    dirs = np.array(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
    )
    picked = farthest_point_sampling(dirs, seed_idx=0, n=3)
    assert sorted(picked) == [0, 1, 2]


def test_fps_empty_input_returns_empty():
    assert farthest_point_sampling(np.empty((0, 3), dtype=np.float32), 0, 4) == []


# ---------------------------------------------------------------------------
# cluster_masks
# ---------------------------------------------------------------------------


def _two_blobs(n_big: int = 30, n_small: int = 10) -> np.ndarray:
    """Two tight point blobs 10 m apart (eps=1.0 separates them cleanly)."""
    rng = np.random.default_rng(0)
    big = rng.normal(0.0, 0.05, (n_big, 3))
    small = rng.normal(10.0, 0.05, (n_small, 3))
    return np.vstack([big, small]).astype(np.float32)


def test_cluster_masks_separates_blobs_largest_first():
    pts = _two_blobs(30, 10)
    masks = cluster_masks(pts, eps=1.0, min_samples=3)

    assert len(masks) == 2
    assert masks[0].sum() == 30 and masks[1].sum() == 10   # largest first
    # Masks are disjoint and each covers exactly one blob.
    assert not np.any(masks[0] & masks[1])
    assert masks[0][:30].all() and masks[1][30:].all()


def test_cluster_masks_min_size_drops_small_cluster():
    pts = _two_blobs(30, 10)
    masks = cluster_masks(pts, eps=1.0, min_samples=3, min_size=20)

    assert len(masks) == 1
    assert masks[0].sum() == 30


def test_cluster_masks_all_noise_returns_empty():
    # Points too far apart for eps and min_samples → everything is noise.
    pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
    assert cluster_masks(pts, eps=0.1, min_samples=2) == []


def test_cluster_masks_empty_input():
    assert cluster_masks(np.empty((0, 3), dtype=np.float32)) == []


def test_largest_cluster_matches_first_cluster_mask():
    pts = _two_blobs(30, 10)
    mask = largest_cluster(pts, eps=1.0, min_samples=3)
    masks = cluster_masks(pts, eps=1.0, min_samples=3)
    assert np.array_equal(mask, masks[0])


def test_largest_cluster_keeps_all_noise_fallback():
    pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
    mask = largest_cluster(pts, eps=0.1, min_samples=2)
    assert mask.all()   # no cluster found — keep everything
