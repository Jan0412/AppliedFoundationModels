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
    return_inverse: bool = False,
):
    """Keep one point per occupied voxel (uniform density downsample).

    Besides bounding cost, this evens out density before clustering: an object
    seen from many frames collapses to one dense connected region per voxel,
    while scattered background stays sparse.

    Args:
        points:         ``(N, 3)`` XYZ.
        colors:         Optional ``(N, 3)`` colors, kept in lockstep with
                        *points*.
        voxel:          Edge length in metres. ``<= 0`` disables downsampling.
        return_inverse: Also return an ``(N,)`` index mapping every *input*
                        point to the row of the kept point that represents its
                        voxel. Lets callers carry per-point provenance (e.g.
                        which frame a point came from) across the downsample,
                        which the kept points alone cannot express — each voxel
                        keeps only one arbitrary source point.

    Returns:
        ``(points, colors)`` reduced to one sample per voxel (order arbitrary),
        or ``(points, colors, inverse)`` when *return_inverse* is set.
    """
    points = np.asarray(points)
    if voxel is None or voxel <= 0 or len(points) == 0:
        inverse = np.arange(len(points))
        return (points, colors, inverse) if return_inverse else (points, colors)
    keys = np.floor(points / voxel).astype(np.int64)
    # `idx` picks one representative per unique voxel; `inverse` indexes into
    # that same unique ordering, so it addresses rows of the kept arrays.
    _, idx, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    ds_points = points[idx]
    ds_colors = None if colors is None else np.asarray(colors)[idx]
    if not return_inverse:
        return ds_points, ds_colors
    return ds_points, ds_colors, np.asarray(inverse).reshape(-1)


def cluster_masks(
    points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 10,
    min_size: int = 1,
) -> list[np.ndarray]:
    """Boolean masks of all DBSCAN clusters in *points*, largest first.

    DBSCAN groups dense regions and labels sparse points as noise. Objects —
    consistent across frames — form dense clusters, while per-frame background
    is scattered and falls to noise. Spatially separated instances of the same
    category (e.g. several chairs) come back as separate clusters.

    Args:
        points:      ``(N, 3)`` XYZ.
        eps:         Neighbourhood radius in metres.
        min_samples: Core-point neighbour count.
        min_size:    Drop clusters with fewer points than this.

    Returns:
        One ``(N,)`` boolean mask per kept cluster, sorted by point count
        descending (most multi-frame consensus first). Empty list when DBSCAN
        finds no cluster (every point labelled noise) or *points* is empty.
    """
    from sklearn.cluster import DBSCAN

    points = np.asarray(points)
    if len(points) == 0:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    valid = labels[labels >= 0]
    if valid.size == 0:
        return []
    counts = np.bincount(valid)
    # Stable sort so equal-size clusters keep ascending-label order (same
    # tie-break as the old argmax-based largest_cluster).
    order = np.argsort(-counts, kind="stable")
    return [labels == int(lab) for lab in order if counts[lab] >= min_size]


def largest_cluster(
    points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 10,
) -> np.ndarray:
    """Boolean mask of the largest DBSCAN cluster in *points*.

    Thin wrapper over :func:`cluster_masks` keeping only the dominant cluster.

    Args:
        points:      ``(N, 3)`` XYZ.
        eps:         Neighbourhood radius in metres.
        min_samples: Core-point neighbour count.

    Returns:
        ``(N,)`` boolean mask selecting the largest cluster. Falls back to
        all-True when DBSCAN finds no cluster (every point labelled noise), so
        the caller always gets a usable box.
    """
    points = np.asarray(points)
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)

    masks = cluster_masks(points, eps=eps, min_samples=min_samples)
    if not masks:
        return np.ones(n, dtype=bool)   # no cluster found — keep everything
    return masks[0]


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


def central_depth(
    depth_path: str | Path,
    depth_scale: float,
    patch_frac: float = 0.2,
) -> Optional[float]:
    """Robust metric depth at the centre of a frame.

    Reads the 16-bit depth PNG (same convention as :func:`backproject`) and
    returns the median of the valid (``> 0``) depths inside a centred patch
    whose edges are ``patch_frac`` of the image edges. The median makes the
    estimate robust to depth holes and thin foreground clutter.

    Args:
        depth_path:  Path to the (16-bit) depth PNG, in raw units.
        depth_scale: Divisor converting raw depth units to metres.
        patch_frac:  Patch edge length as a fraction of the image edge
                     (clamped to at least 1 pixel).

    Returns:
        Median depth in metres, or ``None`` when the patch holds no valid
        depth pixels.
    """
    depth = np.asarray(Image.open(depth_path), dtype=np.float32) / float(depth_scale)
    h, w = depth.shape
    ph = max(1, int(h * patch_frac))
    pw = max(1, int(w * patch_frac))
    y0 = (h - ph) // 2
    x0 = (w - pw) // 2
    patch = depth[y0:y0 + ph, x0:x0 + pw]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def lookat_point(cam2world: np.ndarray, depth_m: float) -> np.ndarray:
    """World-space point a camera is looking at.

    Approximates the observed surface point along the optical axis::

        lookat = camera_center + optical_axis * depth_m

    where ``camera_center = cam2world[:3, 3]`` and ``optical_axis =
    cam2world[:3, 2]`` — the world-space camera +Z, the same axis
    :func:`backproject` treats as depth.

    Args:
        cam2world: ``(4, 4)`` camera-to-world pose.
        depth_m:   Distance along the optical axis in metres (e.g. from
                   :func:`central_depth`).

    Returns:
        ``(3,)`` ``float32`` world XYZ.
    """
    M = np.asarray(cam2world, dtype=np.float32)
    return (M[:3, 3] + M[:3, 2] * float(depth_m)).astype(np.float32)


def farthest_point_sampling(
    directions: np.ndarray,
    seed_idx: int,
    n: int,
) -> list[int]:
    """Greedy k-center selection on angular distance between unit vectors.

    Starting from ``seed_idx``, repeatedly adds the vector with the largest
    angle to its nearest already-selected vector — the classic 2-approximation
    of the max-min dispersion (p-dispersion) objective. Duplicated directions
    are never picked twice while distinct ones remain.

    Args:
        directions: ``(N, 3)`` unit vectors.
        seed_idx:   Index of the first selected vector.
        n:          Number of vectors to select.

    Returns:
        ``min(n, N)`` indices into *directions*, seed first.
    """
    dirs = np.asarray(directions, dtype=np.float32)
    n_total = len(dirs)
    if n_total == 0 or n <= 0:
        return []

    selected = [int(seed_idx)]
    # Angle of every vector to its nearest selected vector.
    min_angle = np.arccos(np.clip(dirs @ dirs[seed_idx], -1.0, 1.0))
    min_angle[seed_idx] = -np.inf

    while len(selected) < min(n, n_total):
        nxt = int(np.argmax(min_angle))
        selected.append(nxt)
        angles = np.arccos(np.clip(dirs @ dirs[nxt], -1.0, 1.0))
        min_angle = np.minimum(min_angle, angles)
        min_angle[nxt] = -np.inf

    return selected


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
