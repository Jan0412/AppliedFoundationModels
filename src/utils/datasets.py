"""Dataset adapters: turn an on-disk RGB-D sequence into a uniform bundle.

The indexing/query pipeline is dataset-agnostic — it only consumes RGB paths,
parallel depth paths, parallel 4x4 camera poses, and per-collection intrinsics
+ depth_scale. The *only* dataset-specific code (parsing those out of whatever
layout a benchmark uses) lives here, so swapping datasets is one call::

    fs = load_tum(data_dir, intrinsics={...})     # or: load_scannet(data_dir)
    idx.insert(fs.paths, COLLECTION, ids=...,
               depth_paths=fs.depth_paths, poses=fs.poses,
               intrinsics=fs.intrinsics, depth_scale=fs.depth_scale)

:func:`load_collection` closes the loop: it rebuilds the same bundle back out
of an *indexed* collection, so consumers that need every frame of a scene (the
UI's scene point cloud) work off the DB alone — no dataset dir, no knowledge of
which benchmark the collection came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .db import load_collection_meta
from .geometry import quat_to_cam2world


@dataclass
class FrameSet:
    """Uniform RGB-D bundle consumed by :class:`~src.index.Indexer`.

    Attributes:
        paths:       RGB image paths.
        depth_paths: Parallel depth-map paths (one per RGB frame).
        poses:       Parallel ``(4, 4)`` cam-to-world matrices.
        intrinsics:  Per-collection ``{"fx", "fy", "cx", "cy"}``.
        depth_scale: Divisor converting raw depth units to metres.
    """

    paths: list[str]
    depth_paths: list[str]
    poses: list[np.ndarray]
    intrinsics: dict
    depth_scale: float

    def __len__(self) -> int:
        return len(self.paths)


def _parse_tum_table(path: Path) -> list[list[str]]:
    """Parse a TUM timestamp-file table, skipping comment/blank lines."""
    rows: list[list[str]] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        rows.append(line.split())
    return rows


def load_tum(
    data_dir: str | Path,
    intrinsics: dict,
    depth_scale: float = 5000.0,
    max_dt: float = 0.05,
) -> FrameSet:
    """Load a TUM RGB-D sequence into a :class:`FrameSet`.

    TUM stores RGB frames, depth frames, and ground-truth poses in separate
    files with independent timestamps. Frames are associated by nearest
    timestamp (within *max_dt* seconds), matching the DinoSAM notebook.

    Args:
        data_dir:    Sequence root containing ``rgb.txt``, ``depth.txt`` and
                     ``groundtruth.txt`` plus the ``rgb/`` and ``depth/`` dirs.
        intrinsics:  Camera intrinsics ``{"fx", "fy", "cx", "cy"}`` (TUM
                     intrinsics differ per sequence, so they're supplied).
        depth_scale: Raw-depth → metres divisor (5000 for TUM).
        max_dt:      Maximum timestamp gap (s) for a valid association.

    Returns:
        A :class:`FrameSet` with one entry per associated RGB frame.
    """
    data_dir = Path(data_dir)
    rgb_rows = _parse_tum_table(data_dir / "rgb.txt")
    depth_rows = _parse_tum_table(data_dir / "depth.txt")
    gt_rows = _parse_tum_table(data_dir / "groundtruth.txt")

    depth_ts = np.array([float(r[0]) for r in depth_rows])
    gt_ts = np.array([float(r[0]) for r in gt_rows])

    paths: list[str] = []
    depth_paths: list[str] = []
    poses: list[np.ndarray] = []

    for r in rgb_rows:
        ts = float(r[0])
        di = int(np.argmin(np.abs(depth_ts - ts)))
        gi = int(np.argmin(np.abs(gt_ts - ts)))
        if abs(depth_ts[di] - ts) > max_dt or abs(gt_ts[gi] - ts) > max_dt:
            continue
        paths.append(str(data_dir / r[1]))
        depth_paths.append(str(data_dir / depth_rows[di][1]))
        poses.append(quat_to_cam2world(*gt_rows[gi][1:8]))

    return FrameSet(
        paths=paths,
        depth_paths=depth_paths,
        poses=poses,
        intrinsics=dict(intrinsics),
        depth_scale=depth_scale,
    )


def _natural_key(path: str) -> tuple:
    """Sort key ordering ``frame2`` before ``frame10`` (digit runs compare numerically)."""
    return tuple(
        int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", path)
    )


def load_collection(db, collection_id: str) -> FrameSet:
    """Rebuild a :class:`FrameSet` from an *indexed* collection in LanceDB.

    The rows written by :class:`~src.index.Indexer` already carry everything a
    :class:`FrameSet` needs — ``path``, ``depth_path`` and the flattened
    ``cam2world`` per frame — plus the collection's calibration row. Reading
    them back gives whole-scene access (e.g. for
    :func:`~src.utils.geometry.build_scene_cloud`) without knowing which
    dataset the collection came from, or where that dataset lives on disk.

    Frames are ordered naturally by RGB path, so the result is stable across
    calls (and temporal for datasets naming frames by index or timestamp).
    Rows without a path/depth path, or with a non-finite pose, are skipped —
    mirroring :func:`load_scannet`'s handling of lost-tracking frames.

    Args:
        db:            An open :class:`lancedb.DBConnection`.
        collection_id: Name of the collection (= LanceDB table).

    Returns:
        A :class:`FrameSet` covering every usable frame in the collection.

    Raises:
        ValueError: If the table or its calibration row is missing — without
            intrinsics the collection cannot be back-projected.
    """
    if collection_id not in db.list_tables().tables:
        raise ValueError(f"collection '{collection_id}' is not in this store.")

    meta = load_collection_meta(db, collection_id)
    if meta is None:
        raise ValueError(
            f"collection '{collection_id}' has no calibration row "
            f"(intrinsics + depth_scale); it cannot be back-projected."
        )

    table = db.open_table(collection_id)
    n_rows = table.count_rows()
    rows = (
        table.search()
             .select(["path", "depth_path", "cam2world"])
             .limit(n_rows)
             .to_list()
        if n_rows else []
    )
    rows.sort(key=lambda r: _natural_key(r.get("path") or ""))

    paths: list[str] = []
    depth_paths: list[str] = []
    poses: list[np.ndarray] = []
    for r in rows:
        path       = r.get("path") or ""
        depth_path = r.get("depth_path") or ""
        cam        = r.get("cam2world")
        if not path or not depth_path or cam is None or len(cam) != 16:
            continue
        pose = np.asarray(cam, dtype=np.float32).reshape(4, 4)
        if not np.isfinite(pose).all():
            continue
        paths.append(path)
        depth_paths.append(depth_path)
        poses.append(pose)

    return FrameSet(
        paths=paths,
        depth_paths=depth_paths,
        poses=poses,
        intrinsics={k: meta[k] for k in ("fx", "fy", "cx", "cy")},
        depth_scale=meta["depth_scale"],
    )


def load_scannet(data_dir: str | Path, depth_scale: float = 1000.0) -> FrameSet:
    """Load a ScanNet sequence into a :class:`FrameSet`.

    ScanNet frames are index-aligned (no timestamp association needed): the
    ``i``-th colour, depth, and pose files share the same frame number. Poses
    are 4x4 cam-to-world matrices; intrinsics come from
    ``intrinsic/intrinsic_depth.txt``.

    Expected layout::

        data_dir/
          color/{i}.jpg
          depth/{i}.png
          pose/{i}.txt
          intrinsic/intrinsic_depth.txt

    Args:
        data_dir:    Sequence root.
        depth_scale: Raw-depth → metres divisor (1000 for ScanNet).

    Returns:
        A :class:`FrameSet` ordered by frame index.

    Note:
        Implemented to the documented ScanNet layout but not verified against
        live ScanNet data; covered by a synthetic-fixture unit test.
    """
    data_dir = Path(data_dir)
    color_dir = data_dir / "color"
    depth_dir = data_dir / "depth"
    pose_dir = data_dir / "pose"

    # Frame indices present as colour images, ordered numerically.
    indices = sorted(
        (int(p.stem) for p in color_dir.glob("*.jpg")),
    )

    K = np.loadtxt(data_dir / "intrinsic" / "intrinsic_depth.txt").reshape(4, 4)
    intrinsics = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }

    paths: list[str] = []
    depth_paths: list[str] = []
    poses: list[np.ndarray] = []
    for i in indices:
        pose = np.loadtxt(pose_dir / f"{i}.txt").reshape(4, 4).astype(np.float32)
        # ScanNet marks lost-tracking frames with inf/nan poses — skip them.
        if not np.isfinite(pose).all():
            continue
        paths.append(str(color_dir / f"{i}.jpg"))
        depth_paths.append(str(depth_dir / f"{i}.png"))
        poses.append(pose)

    return FrameSet(
        paths=paths,
        depth_paths=depth_paths,
        poses=poses,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
    )
