"""Shared pytest fixtures for tests/index/.

Fixtures
--------
fake_pil_image    – a tiny 32×32 RGB PIL image (no disk I/O).
make_image_files  – factory that writes N tiny PNGs into a tmp dir and returns
                    their paths.
mock_siglip_model – a SigLIPModel-shaped object with batch_size, embedding_dim
                    and embed_images mocked.  No HuggingFace model is loaded.
tmp_db_path       – temp directory for an isolated LanceDB store.
indexer           – fully constructed Indexer wired to the mock model + tmp DB.
tmp_indexing_config – temp config.yaml containing both `models:` and
                    `indexing:` sections, pointing at the tmp DB.

The indexer uses real LanceDB on a temp directory — no DB mocking — so the
tests also catch any drift in the (well-supported) LanceDB API.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from PIL import Image

from src.index import Indexer


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_pil_image() -> Image.Image:
    """Tiny 32×32 solid-colour RGB PIL image."""
    return Image.new("RGB", (32, 32), color=(100, 150, 200))


@pytest.fixture
def make_image_files(tmp_path):
    """Factory that materialises N tiny PNGs and returns their string paths."""

    def _make(n: int, subdir: str = "imgs") -> list[str]:
        d = tmp_path / subdir
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            p = d / f"{i:04d}.png"
            Image.new("RGB", (32, 32), color=(i * 5 % 255, 0, 0)).save(p)
            paths.append(str(p))
        return paths

    return _make


# ---------------------------------------------------------------------------
# Mock SigLIPModel (Indexer only uses batch_size, embedding_dim, embed_images)
# ---------------------------------------------------------------------------


EMBED_DIM = 8


def _normalized_rows(n: int, dim: int = EMBED_DIM, seed: int = 0) -> np.ndarray:
    """Return an (n, dim) float32 array with each row L2-normalised."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@pytest.fixture
def mock_siglip_model() -> MagicMock:
    """SigLIPModel-shaped mock with deterministic embed_images."""
    model = MagicMock(name="SigLIPModel")
    model.batch_size = 8
    model.embedding_dim = EMBED_DIM

    def _embed(pil_images):
        assert isinstance(pil_images, list)
        return _normalized_rows(len(pil_images), EMBED_DIM)

    model.embed_images = MagicMock(side_effect=_embed)
    return model


# ---------------------------------------------------------------------------
# LanceDB
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path(tmp_path) -> Path:
    """Path to a fresh LanceDB directory for one test."""
    return tmp_path / "lancedb"


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


@pytest.fixture
def indexer(mock_siglip_model, tmp_db_path) -> Indexer:
    """Indexer with the mock model and a small batch_size to exercise batching."""
    return Indexer(model=mock_siglip_model, db_path=tmp_db_path, batch_size=2)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_indexing_config(tmp_path) -> Path:
    """A config.yaml with models: + indexing: pointing at a tmp DB path."""
    cfg = {
        "models": {
            "siglip": {
                "model_id": "google/siglip2-base-patch16-224",
                "device": "auto",
                "batch_size": 64,
            },
            "sam": {
                "model_id": "facebook/sam3",
                "device": "auto",
                "threshold": 0.5,
                "mask_threshold": 0.5,
            },
        },
        "indexing": {
            "db_path": str(tmp_path / "lancedb_from_cfg"),
            "batch_size": 4,
        },
    }
    f = tmp_path / "config.yaml"
    f.write_text(yaml.dump(cfg))
    return f
