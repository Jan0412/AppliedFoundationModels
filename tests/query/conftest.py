"""Shared fixtures for tests/query/.

Provides:
- ``embed_dim``           — small embedding width used everywhere here.
- ``mock_siglip_model``   — SigLIPModel-shaped mock; only ``embed_text`` is wired.
- ``mock_sam_model``      — SAMModel-shaped mock returning masks + boxes + scores.
- ``mock_dino_model``     — GroundingDINOModel-shaped mock returning labels + boxes + scores.
- ``tiny_image_files``    — factory that writes N tiny PNGs to disk.
- ``populated_db``        — real LanceDB store containing a 1-table collection
                            with known vectors + image paths, plus the connection.
- ``detected_factory``    — builds list[DetectedImage] with given scores, for
                            the rerank step's pure-logic tests.
- ``pose_looking_at``     — 4x4 cam2world builder whose +Z column aims at a target.
- ``db_with_sims``        — factory for a real LanceDB collection with exact
                            cosine similarities and circle-pose geometry.
- ``pool_from``           — factory for a retrieval pool (images unloaded)
                            with circle poses + depth maps, no DB involved.
- ``meta_db``             — factory for a store holding only ``_collection_meta``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest
import torch
from PIL import Image

from src.data_model import DetectedImage, RetrievedImage
from src.utils.db import connect


EMBED_DIM = 4


@pytest.fixture
def embed_dim() -> int:
    return EMBED_DIM


@pytest.fixture
def mock_siglip_model() -> MagicMock:
    """SigLIPModel-shaped mock — only ``embed_text`` is exercised."""
    model = MagicMock(name="SigLIPModel")
    model.embedding_dim = EMBED_DIM

    def _embed_text(text: str) -> np.ndarray:
        v = np.zeros(EMBED_DIM, dtype=np.float32)
        v[0] = 1.0
        return v

    model.embed_text = MagicMock(side_effect=_embed_text)
    return model


@pytest.fixture
def mock_sam_model() -> MagicMock:
    """SAMModel-shaped mock. Returns ``{masks, boxes, scores}`` (no labels).

    Tests can override ``model.invoke.side_effect`` to return per-call data.
    """
    model = MagicMock(name="SAMModel")

    def _default_invoke(inp):  # noqa: ARG001
        return {
            "masks": [torch.zeros(8, 8, dtype=torch.bool)],
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "scores": torch.tensor([0.5]),
        }

    model.invoke = MagicMock(side_effect=_default_invoke)
    return model


@pytest.fixture
def mock_dino_model() -> MagicMock:
    """GroundingDINOModel-shaped mock. Returns ``{labels, boxes, scores}`` (no masks)."""
    model = MagicMock(name="GroundingDINOModel")

    def _default_invoke(inp):  # noqa: ARG001
        return {
            "labels": ["object"],
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "scores": torch.tensor([0.5]),
        }

    model.invoke = MagicMock(side_effect=_default_invoke)
    return model


@pytest.fixture
def tiny_image_files(tmp_path):
    def _make(n: int) -> list[str]:
        d = tmp_path / "imgs"
        d.mkdir(exist_ok=True)
        paths = []
        for i in range(n):
            p = d / f"{i:02d}.png"
            Image.new("RGB", (8, 8), color=(i * 30 % 255, 0, 0)).save(p)
            paths.append(str(p))
        return paths

    return _make


@pytest.fixture
def populated_db(tmp_path, tiny_image_files):
    """Create a LanceDB collection ``coll`` with 3 known vectors + images.

    The collection is fully projectable: each row carries a ``depth_path`` and
    a flattened identity ``cam2world`` pose, and a ``_collection_meta`` row holds
    the intrinsics + depth_scale (mirroring what :class:`Indexer` writes).

    Returns a dict with:
        db             — open lancedb.DBConnection
        collection_id  — "coll"
        ids            — [str, str, str]
        paths          — [str, str, str]
        depth_paths    — [str, str, str]
        vectors        — np.ndarray, shape (3, EMBED_DIM), L2-normalised
        intrinsics     — {"fx","fy","cx","cy"}
        depth_scale    — float
    """
    paths = tiny_image_files(3)
    # Constant-depth 8x8 16-bit depth maps, one per RGB frame.
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir(exist_ok=True)
    depth_paths = []
    for i in range(3):
        dp = depth_dir / f"{i:02d}.png"
        Image.fromarray(np.full((8, 8), 1000, dtype=np.uint16)).save(dp)
        depth_paths.append(str(dp))

    # Three normalised, axis-aligned vectors so cosine similarity to
    # [1,0,0,0] gives a clean ranking: 1.0, 0.0, 0.0.
    vectors = np.eye(3, EMBED_DIM, dtype=np.float32)
    identity16 = np.eye(4, dtype=np.float32).reshape(16).tolist()

    db_dir = tmp_path / "lancedb"
    db = connect(db_dir)
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("collection_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
        pa.field("path", pa.string()),
        pa.field("depth_path", pa.string()),
        pa.field("cam2world", pa.list_(pa.float32())),
    ])
    table = db.create_table("coll", schema=schema)
    rows = [
        {
            "id": f"id-{i}",
            "collection_id": "coll",
            "vector": vectors[i].tolist(),
            "path": paths[i],
            "depth_path": depth_paths[i],
            "cam2world": identity16,
        }
        for i in range(3)
    ]
    table.add(rows)
    ids = [r["id"] for r in rows]

    intrinsics = {"fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 4.0}
    depth_scale = 1000.0
    meta_schema = pa.schema([
        pa.field("collection_id", pa.string()),
        pa.field("fx", pa.float64()),
        pa.field("fy", pa.float64()),
        pa.field("cx", pa.float64()),
        pa.field("cy", pa.float64()),
        pa.field("depth_scale", pa.float64()),
    ])
    meta = db.create_table("_collection_meta", schema=meta_schema)
    meta.add([{"collection_id": "coll", **intrinsics, "depth_scale": depth_scale}])

    return {
        "db": db,
        "collection_id": "coll",
        "ids": ids,
        "paths": paths,
        "depth_paths": depth_paths,
        "vectors": vectors,
        "intrinsics": intrinsics,
        "depth_scale": depth_scale,
    }


@pytest.fixture
def detected_factory():
    """Build a list[DetectedImage] with the given detection_scores."""

    def _make(scores: list[float]) -> list[DetectedImage]:
        out = []
        for i, s in enumerate(scores):
            out.append(
                DetectedImage(
                    id=f"id-{i}",
                    path=f"/tmp/{i}.png",
                    similarity_score=0.5,
                    detection_score=s,
                    boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    scores=torch.tensor([s]),
                    masks=[torch.zeros(2, 2, dtype=torch.bool)],
                )
            )
        return out

    return _make


@pytest.fixture
def retrieved_from(tiny_image_files):
    """Build a list[RetrievedImage] of length n with fresh PIL images."""

    def _make(n: int) -> list[RetrievedImage]:
        paths = tiny_image_files(n)
        return [
            RetrievedImage(
                id=f"id-{i}",
                path=paths[i],
                similarity_score=1.0 - i * 0.1,
                image=Image.open(paths[i]).convert("RGB"),
            )
            for i in range(n)
        ]

    return _make


def _pose_looking_at(cam_pos, target) -> np.ndarray:
    """4x4 cam2world whose optical axis (+Z column) points at *target*."""
    cam_pos = np.asarray(cam_pos, dtype=np.float32)
    z = np.asarray(target, dtype=np.float32) - cam_pos
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(up, z))) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4, dtype=np.float32)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, cam_pos
    return M


@pytest.fixture
def pose_looking_at():
    """Expose the pose-builder helper to tests."""
    return _pose_looking_at


def _circle_positions(n: int) -> list[np.ndarray]:
    """n camera positions evenly spaced on the unit circle in the XY plane."""
    return [
        np.array([np.cos(a), np.sin(a), 0.0], dtype=np.float32)
        for a in 2.0 * np.pi * np.arange(n) / max(n, 1)
    ]


def _write_depths(dir_path: Path, values: list[int], size: int = 8) -> list[str]:
    """One constant-value 16-bit depth PNG per entry in *values*."""
    dir_path.mkdir(parents=True, exist_ok=True)
    out = []
    for i, v in enumerate(values):
        p = dir_path / f"{i:03d}.png"
        Image.fromarray(np.full((size, size), v, dtype=np.uint16)).save(p)
        out.append(str(p))
    return out


@pytest.fixture
def db_with_sims(tmp_path, tiny_image_files):
    """Factory: real LanceDB collection with exact cosine similarities + geometry.

    ``_make(sims)`` writes one row per similarity ``s`` with vector
    ``[s, sqrt(1-s²), 0, 0]`` — cosine to the mock query ``[1,0,0,0]`` is
    exactly ``s``. Each row gets a constant 1.0 m depth map and a pose on the
    unit circle looking at the origin (so every look-at point is the origin
    and viewing directions are the radial directions). ``_collection_meta``
    is written like the indexer would.
    """

    def _make(sims: list[float], collection_id: str = "dyncoll") -> dict:
        n = len(sims)
        paths = tiny_image_files(n)
        depth_paths = _write_depths(tmp_path / f"depth_{collection_id}", [1000] * n)
        positions = _circle_positions(n)

        db = connect(tmp_path / f"lancedb_{collection_id}")
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("collection_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
            pa.field("path", pa.string()),
            pa.field("depth_path", pa.string()),
            pa.field("cam2world", pa.list_(pa.float32())),
        ])
        table = db.create_table(collection_id, schema=schema)
        rows = [
            {
                "id": f"id-{i}",
                "collection_id": collection_id,
                "vector": [float(s), float(np.sqrt(1.0 - s * s)), 0.0, 0.0],
                "path": paths[i],
                "depth_path": depth_paths[i],
                "cam2world": _pose_looking_at(positions[i], (0, 0, 0))
                .reshape(16)
                .tolist(),
            }
            for i, s in enumerate(sims)
        ]
        if rows:
            table.add(rows)

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
            "fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 4.0,
            "depth_scale": 1000.0,
        }])

        return {
            "db": db,
            "collection_id": collection_id,
            "ids": [r["id"] for r in rows],
            "paths": paths,
            "depth_paths": depth_paths,
            "sims": np.asarray(sims, dtype=np.float32),
        }

    return _make


@pytest.fixture
def pool_from(tmp_path, tiny_image_files):
    """Factory: list[RetrievedImage] pool (images unloaded) with real geometry.

    ``_make(sims)`` places cameras on the unit circle looking at the origin
    and writes a constant 1.0 m depth PNG per frame. ``depth_values`` allows
    per-frame overrides (raw uint16 units, scale 1000). Frames are returned
    in the given order — pass *sims* descending to mimic retrieval output.
    """

    counter = {"i": 0}

    def _make(sims: list[float], depth_values: list[int] | None = None) -> list[RetrievedImage]:
        n = len(sims)
        counter["i"] += 1
        paths = tiny_image_files(n)
        depth_paths = _write_depths(
            tmp_path / f"pool_depth_{counter['i']}", depth_values or [1000] * n
        )
        positions = _circle_positions(n)
        return [
            RetrievedImage(
                id=f"id-{i}",
                path=paths[i],
                similarity_score=float(sims[i]),
                image=None,
                depth_path=depth_paths[i],
                cam2world=_pose_looking_at(positions[i], (0, 0, 0)),
            )
            for i in range(n)
        ]

    return _make


@pytest.fixture
def meta_db(tmp_path):
    """Factory: LanceDB store holding only a ``_collection_meta`` row."""

    counter = {"i": 0}

    def _make(collection_id: str = "coll", depth_scale: float = 1000.0):
        counter["i"] += 1
        db = connect(tmp_path / f"metadb_{collection_id}_{counter['i']}")
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
            "fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 4.0,
            "depth_scale": depth_scale,
        }])
        return db

    return _make
