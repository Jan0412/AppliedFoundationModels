"""ScanRefer evaluation helpers: GT-box extraction, IoU, and eval-item building.

All geometry is in the **raw ScanNet scan frame** (do NOT apply axisAlignment).
Predicted boxes from the pipeline live in this same frame (verified empirically:
99% of back-projected points fall inside the mesh AABB without any alignment).
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Normalize an object label: underscore→space, strip, lowercase."""
    return name.replace("_", " ").strip().lower()


def read_ply_xyz(path: str | Path) -> np.ndarray:
    """Parse vertex XYZ from a binary-little-endian PLY file (no dependencies).

    Reads only the vertex element; ignores face/edge elements and all per-vertex
    properties other than ``x``, ``y``, ``z``.

    Returns:
        ``(N, 3)`` float64 array of vertex positions.
    """
    _typemap: dict[bytes, str] = {
        b"float": "f4", b"float32": "f4", b"double": "f8",
        b"uchar": "u1", b"uint8": "u1",
        b"int": "i4", b"uint": "u4",
        b"short": "i2", b"ushort": "u2",
    }

    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply", "not a PLY file"
        f.readline()  # format line
        n_vert: int | None = None
        props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            ln = f.readline().split()
            if ln[0] == b"element":
                in_vertex = ln[1] == b"vertex"
                if in_vertex:
                    n_vert = int(ln[2])
            elif ln[0] == b"property" and in_vertex:
                props.append((ln[2].decode(), _typemap[ln[1]]))
            elif ln[0] == b"end_header":
                break

        if n_vert is None:
            raise ValueError(f"PLY has no vertex element: {path}")

        dt = np.dtype(props)
        data = np.frombuffer(f.read(n_vert * dt.itemsize), dtype=dt, count=n_vert)

    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


def scene_class_counts(scene_dir: str | Path) -> Counter:
    """Return normalized label → instance count from a ScanNet aggregation file.

    Uses ``<scene>_vh_clean.aggregation.json`` (not the ``_vh_clean_2`` variant,
    which is the same data but at higher mesh resolution — both have the same
    ``segGroups`` content for label/segment purposes).
    """
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name
    agg_path = scene_dir / f"{scene_id}.aggregation.json"
    agg = json.loads(agg_path.read_text())
    return Counter(_norm(g["label"]) for g in agg["segGroups"])


def gt_box(scene_dir: str | Path, object_id: int) -> np.ndarray:
    """Compute the ground-truth AABB for *object_id* in the raw scan frame.

    Resolves object_id → segment ids (aggregation.json) → vertex indices
    (segs.json) → mesh vertices (vh_clean_2.ply) → axis-aligned bounding box.

    Args:
        scene_dir: Root directory of the ScanNet scene.
        object_id: Integer object id from the ScanRefer annotation.

    Returns:
        ``(2, 3)`` array ``[[xmin,ymin,zmin], [xmax,ymax,zmax]]`` in metres,
        raw scan frame (no axisAlignment applied).

    Raises:
        ValueError: If *object_id* is not found in the aggregation.
    """
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name

    agg = json.loads((scene_dir / f"{scene_id}.aggregation.json").read_text())
    groups = {g["objectId"]: g for g in agg["segGroups"]}
    if object_id not in groups:
        raise ValueError(
            f"object_id={object_id} not found in {scene_id}.aggregation.json"
        )
    target_segs = set(groups[object_id]["segments"])

    seg_indices = json.loads(
        (scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json").read_text()
    )["segIndices"]
    seg_arr = np.asarray(seg_indices, dtype=np.int32)
    vertex_mask = np.isin(seg_arr, list(target_segs))

    xyz = read_ply_xyz(scene_dir / f"{scene_id}_vh_clean_2.ply")
    pts = xyz[vertex_mask]

    if len(pts) == 0:
        raise ValueError(
            f"object_id={object_id} in {scene_id} resolved to zero mesh vertices"
        )

    return np.stack([pts.min(axis=0), pts.max(axis=0)]).astype(np.float32)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou_aabb(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """3D axis-aligned IoU of two ``(2,3)`` bounding boxes.

    Args:
        a: ``[[xmin,ymin,zmin],[xmax,ymax,zmax]]`` or ``None``.
        b: Same shape, or ``None``.

    Returns:
        Float in ``[0, 1]``. Returns ``0.0`` when either box is ``None``,
        degenerate, or non-overlapping.
    """
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    inter_min = np.maximum(a[0], b[0])
    inter_max = np.minimum(a[1], b[1])
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(inter_dims[0] * inter_dims[1] * inter_dims[2])
    if inter_vol == 0.0:
        return 0.0

    def _vol(box: np.ndarray) -> float:
        d = np.maximum(box[1] - box[0], 0.0)
        return float(d[0] * d[1] * d[2])

    union_vol = _vol(a) + _vol(b) - inter_vol
    if union_vol <= 0.0:
        return 0.0
    return inter_vol / union_vol


# ---------------------------------------------------------------------------
# ScanRefer data loading
# ---------------------------------------------------------------------------

def load_val(scanrefer_dir: str | Path) -> list[dict]:
    """Load all ScanRefer val annotations as a list of dicts.

    Each dict has keys: ``scene_id``, ``object_id`` (str), ``object_name``,
    ``ann_id``, ``description``, ``token``.
    """
    path = Path(scanrefer_dir) / "ScanRefer_filtered_val.json"
    return json.loads(path.read_text())


def val_scene_order(scanrefer_dir: str | Path) -> list[str]:
    """Return val scene ids in the order given by ``ScanRefer_filtered_val.txt``."""
    path = Path(scanrefer_dir) / "ScanRefer_filtered_val.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Eval-item builder
# ---------------------------------------------------------------------------

@dataclass
class EvalItem:
    """One evaluation query together with its ground-truth reference.

    Attributes:
        scene_id:    ScanNet scene identifier.
        object_id:   Integer object id (the GT target).
        object_name: Raw ScanRefer object_name string (underscores, original case).
        query_text:  The text to feed into the pipeline (either ``object_name``
                     or a natural-language description, depending on query mode).
        ann_id:      Annotation id from ScanRefer; ``"dedup"`` for object_name mode.
    """

    scene_id: str
    object_id: int
    object_name: str
    query_text: str
    ann_id: str


def eval_items(
    val: list[dict],
    scenes: list[str],
    scans_root: str | Path,
    query_mode: Literal["object_name", "description"] = "object_name",
) -> list[EvalItem]:
    """Build the list of evaluation items for a given scene subset and query mode.

    Both modes restrict to **unique** targets: objects whose category appears
    exactly once in the scene (so the category name alone identifies the target).

    Args:
        val:        ScanRefer val annotations (from :func:`load_val`).
        scenes:     Ordered list of scene ids to include.
        scans_root: Root directory of the ScanNet scan mirror.
        query_mode: ``"object_name"`` — one item per unique ``(scene_id, object_id)``,
                    query = ``object_name`` (spaces, not underscores).
                    ``"description"`` — one item per val description whose target is
                    unique; query = the full description.

    Returns:
        List of :class:`EvalItem` in scene order.
    """
    if query_mode not in ("object_name", "description"):
        raise ValueError(
            f"query_mode must be 'object_name' or 'description'; got {query_mode!r}"
        )

    scene_set = set(scenes)
    scans_root = Path(scans_root)

    # Pre-compute class counts per scene (one I/O call per scene).
    counts: dict[str, Counter] = {}
    for s in scenes:
        scene_dir = scans_root / s
        if not scene_dir.exists():
            warnings.warn(f"eval_items: scene directory not found, skipping: {scene_dir}")
            continue
        try:
            counts[s] = scene_class_counts(scene_dir)
        except Exception as exc:
            warnings.warn(f"eval_items: could not read class counts for {s}: {exc}")

    items: list[EvalItem] = []

    if query_mode == "object_name":
        seen: set[tuple[str, int]] = set()
        for entry in val:
            s = entry["scene_id"]
            if s not in scene_set or s not in counts:
                continue
            oid = int(entry["object_id"])
            key = (s, oid)
            if key in seen:
                continue
            if counts[s].get(_norm(entry["object_name"]), 0) == 1:
                seen.add(key)
                items.append(EvalItem(
                    scene_id=s,
                    object_id=oid,
                    object_name=entry["object_name"],
                    query_text=_norm(entry["object_name"]),
                    ann_id="dedup",
                ))
    else:  # description
        for entry in val:
            s = entry["scene_id"]
            if s not in scene_set or s not in counts:
                continue
            if counts[s].get(_norm(entry["object_name"]), 0) == 1:
                items.append(EvalItem(
                    scene_id=s,
                    object_id=int(entry["object_id"]),
                    object_name=entry["object_name"],
                    query_text=entry["description"],
                    ann_id=entry["ann_id"],
                ))

    return items
