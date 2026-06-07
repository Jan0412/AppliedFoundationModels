"""Camera geometry helpers for back-projecting 2D masks into 3D world space.

These are the dataset-agnostic primitives behind the query pipeline's
:class:`~src.query.project.ProjectTo3D` step and the dataset adapters in
:mod:`src.utils.datasets`. They are the same operations the DinoSAM notebook
performs by hand, factored out so the pipeline and notebooks share one
implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def quat_to_cam2world(
    tx: float, ty: float, tz: float,
    qx: float, qy: float, qz: float, qw: float,
) -> np.ndarray:
    """Build a 4x4 camera-to-world matrix from a translation + quaternion.

    Matches the TUM ``groundtruth.txt`` convention (``tx ty tz qx qy qz qw``).

    Args:
        tx, ty, tz: Camera translation in world space.
        qx, qy, qz, qw: Orientation quaternion (TUM ordering).

    Returns:
        A ``(4, 4)`` ``float32`` homogeneous transform.
    """
    qx, qy, qz, qw = float(qx), float(qy), float(qz), float(qw)
    tx, ty, tz = float(tx), float(ty), float(tz)
    R = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx ** 2 + qy ** 2)],
    ], dtype=np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


def backproject(
    mask: np.ndarray,
    depth_path: str | Path,
    cam2world: np.ndarray,
    intr: Tuple[float, float, float, float],
    depth_scale: float,
    rgb_path: Optional[str | Path] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Back-project the masked pixels of a frame into world-space 3D points.

    For each pixel ``(u, v)`` inside *mask* with a valid depth ``Z``::

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    then transformed by *cam2world* into world coordinates.

    Args:
        mask:        Boolean array ``(H, W)`` selecting the object's pixels.
        depth_path:  Path to the (16-bit) depth PNG, in raw units.
        cam2world:   ``(4, 4)`` camera-to-world pose.
        intr:        Camera intrinsics ``(fx, fy, cx, cy)``.
        depth_scale: Divisor converting raw depth units to metres.
        rgb_path:    Optional RGB image path; when given, the returned colors
                     are sampled here, otherwise ``colors`` is ``None``.

    Returns:
        ``(points, colors)`` where ``points`` is ``(N, 3)`` ``float32`` world
        XYZ and ``colors`` is ``(N, 3)`` ``uint8`` RGB (or ``None``).
    """
    fx, fy, cx, cy = intr
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(Image.open(depth_path), dtype=np.float32) / float(depth_scale)

    # Masks (SAM or rasterised box) may be at RGB resolution while depth is at
    # sensor resolution.  Resize to depth grid with nearest-neighbour so every
    # pixel maps to a unique depth sample.
    if mask.shape != depth.shape:
        h_d, w_d = depth.shape
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize((w_d, h_d), Image.NEAREST)
        ) > 0

    ys, xs = np.where(mask & (depth > 0))
    if len(xs) == 0:
        empty_pts = np.empty((0, 3), dtype=np.float32)
        return empty_pts, (None if rgb_path is None else np.empty((0, 3), dtype=np.uint8))

    Z = depth[ys, xs]
    pts_c = np.stack(
        [(xs - cx) * Z / fx, (ys - cy) * Z / fy, Z, np.ones_like(Z)], axis=1
    )
    pts_w = (np.asarray(cam2world, dtype=np.float32) @ pts_c.T).T[:, :3]

    colors: Optional[np.ndarray] = None
    if rgb_path is not None:
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        h_rgb, w_rgb = rgb.shape[:2]
        h_d, w_d = depth.shape
        if (h_rgb, w_rgb) != (h_d, w_d):
            # ys, xs are in depth coordinate space; scale to RGB coordinate space.
            xs_rgb = np.clip(np.round(xs * w_rgb / w_d).astype(int), 0, w_rgb - 1)
            ys_rgb = np.clip(np.round(ys * h_rgb / h_d).astype(int), 0, h_rgb - 1)
            colors = rgb[ys_rgb, xs_rgb]
        else:
            colors = rgb[ys, xs]

    return pts_w.astype(np.float32), colors


def box_to_mask(box, height: int, width: int) -> np.ndarray:
    """Rasterise a 2D bounding box into a boolean mask.

    Lets the box-only detection path (e.g. Grounding DINO) reuse
    :func:`backproject`: build a rectangular mask, then back-project it.

    Args:
        box:    ``[x0, y0, x1, y1]`` in pixel coordinates (floats accepted).
        height: Mask height (match the depth image).
        width:  Mask width (match the depth image).

    Returns:
        A ``(height, width)`` boolean array, True inside the (clamped) box.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    xa_i = max(0, int(np.floor(xa)))
    ya_i = max(0, int(np.floor(ya)))
    xb_i = min(width, int(np.ceil(xb)))
    yb_i = min(height, int(np.ceil(yb)))
    mask = np.zeros((height, width), dtype=bool)
    if xb_i > xa_i and yb_i > ya_i:
        mask[ya_i:yb_i, xa_i:xb_i] = True
    return mask


def robust_bounds(
    points: np.ndarray,
    lo: float = 2.0,
    hi: float = 98.0,
) -> Optional[np.ndarray]:
    """Axis-aligned world bounds of *points*, trimmed by per-axis percentiles.

    Using the ``[lo, hi]`` percentile per axis (instead of raw min/max) keeps a
    handful of background or mask-edge depth pixels from inflating the box.

    Args:
        points: ``(N, 3)`` world-space XYZ.
        lo:     Lower percentile (e.g. 2 → drops the closest 2%).
        hi:     Upper percentile (e.g. 98 → drops the farthest 2%).

    Returns:
        ``(2, 3)`` array ``[[xmin,ymin,zmin], [xmax,ymax,zmax]]``, or ``None``
        when *points* is empty.
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return None
    mn = np.percentile(points, lo, axis=0)
    mx = np.percentile(points, hi, axis=0)
    return np.stack([mn, mx]).astype(np.float32)


def voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel: float = 0.02,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Keep one point per occupied voxel (uniform density downsample).

    Besides bounding cost, this evens out density before clustering: an object
    seen from many frames collapses to one dense connected region per voxel,
    while scattered background stays sparse.

    Args:
        points: ``(N, 3)`` XYZ.
        colors: Optional ``(N, 3)`` colors, kept in lockstep with *points*.
        voxel:  Edge length in metres. ``<= 0`` disables downsampling.

    Returns:
        ``(points, colors)`` reduced to one sample per voxel (order arbitrary).
    """
    points = np.asarray(points)
    if voxel is None or voxel <= 0 or len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx], (None if colors is None else np.asarray(colors)[idx])


def largest_cluster(
    points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 10,
) -> np.ndarray:
    """Boolean mask of the largest DBSCAN cluster in *points*.

    DBSCAN groups dense regions and labels sparse points as noise. The object —
    consistent across frames — forms the dominant dense cluster, while per-frame
    background is scattered and falls to noise.

    Args:
        points:      ``(N, 3)`` XYZ.
        eps:         Neighbourhood radius in metres.
        min_samples: Core-point neighbour count.

    Returns:
        ``(N,)`` boolean mask selecting the largest cluster. Falls back to
        all-True when DBSCAN finds no cluster (every point labelled noise), so
        the caller always gets a usable box.
    """
    from sklearn.cluster import DBSCAN

    points = np.asarray(points)
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    valid = labels[labels >= 0]
    if valid.size == 0:
        return np.ones(n, dtype=bool)   # no cluster found — keep everything
    counts = np.bincount(valid)
    return labels == int(counts.argmax())


def aabb_corners(bbox: np.ndarray) -> np.ndarray:
    """Expand a ``(2, 3)`` min/max box into its 8 corner points.

    Convenient for drawing the 12 box edges in a 3D plot.

    Args:
        bbox: ``[[xmin,ymin,zmin], [xmax,ymax,zmax]]``.

    Returns:
        ``(8, 3)`` array of corner coordinates.
    """
    (x0, y0, z0), (x1, y1, z1) = np.asarray(bbox, dtype=np.float32)
    return np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)


def build_scene_cloud(
    fs,
    n_frames: int = 250,
    voxel: float = 0.02,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Back-project N evenly-sampled full frames into a merged scene point cloud.

    Each frame is back-projected completely (all depth-valid pixels), then
    downsampled per-frame before concatenation to bound memory. A second global
    downsample at the same *voxel* size ensures uniform density in the result.

    Args:
        fs:       A :class:`~src.utils.datasets.FrameSet` (or any object with
                  ``paths``, ``depth_paths``, ``poses``, ``intrinsics``,
                  ``depth_scale``).
        n_frames: Maximum number of frames to use (evenly sampled from ``fs``).
        voxel:    Voxel edge length in metres for downsampling (``<= 0`` keeps
                  all points).

    Returns:
        ``(points, colors)`` where ``points`` is ``(M, 3)`` float32 world XYZ
        and ``colors`` is ``(M, 3)`` uint8 RGB (or ``None`` if paths are absent).
    """
    n = len(fs.paths)
    if n == 0:
        return np.empty((0, 3), dtype=np.float32), None

    sample_idx = np.linspace(0, n - 1, min(n_frames, n), dtype=int)
    intr = (
        fs.intrinsics["fx"], fs.intrinsics["fy"],
        fs.intrinsics["cx"], fs.intrinsics["cy"],
    )

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    have_colors = True

    for i in sample_idx:
        # Full-frame mask at depth resolution — no resize needed inside backproject.
        w_d, h_d = Image.open(fs.depth_paths[i]).size
        mask = np.ones((h_d, w_d), dtype=bool)
        pts, cols = backproject(
            mask, fs.depth_paths[i], fs.poses[i], intr, fs.depth_scale,
            rgb_path=fs.paths[i],
        )
        if len(pts) == 0:
            continue
        pts, cols = voxel_downsample(pts, cols, voxel)
        all_pts.append(pts)
        if cols is None:
            have_colors = False
        else:
            all_cols.append(cols)

    if not all_pts:
        return np.empty((0, 3), dtype=np.float32), None

    merged_pts = np.concatenate(all_pts)
    merged_cols = np.concatenate(all_cols) if (have_colors and all_cols) else None
    return voxel_downsample(merged_pts, merged_cols, voxel)
