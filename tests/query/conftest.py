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

    Returns a dict with:
        db             — open lancedb.DBConnection
        collection_id  — "coll"
        ids            — [str, str, str]
        paths          — [str, str, str]
        vectors        — np.ndarray, shape (3, EMBED_DIM), L2-normalised
    """
    paths = tiny_image_files(3)
    # Three normalised, axis-aligned vectors so cosine similarity to
    # [1,0,0,0] gives a clean ranking: 1.0, 0.0, 0.0.
    vectors = np.eye(3, EMBED_DIM, dtype=np.float32)

    db_dir = tmp_path / "lancedb"
    db = connect(db_dir)
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("collection_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
        pa.field("path", pa.string()),
        pa.field("timestamp", pa.float64()),
    ])
    table = db.create_table("coll", schema=schema)
    rows = [
        {
            "id": f"id-{i}",
            "collection_id": "coll",
            "vector": vectors[i].tolist(),
            "path": paths[i],
            "timestamp": 0.0,
        }
        for i in range(3)
    ]
    table.add(rows)
    ids = [r["id"] for r in rows]

    return {
        "db": db,
        "collection_id": "coll",
        "ids": ids,
        "paths": paths,
        "vectors": vectors,
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
