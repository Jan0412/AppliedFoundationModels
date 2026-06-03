"""Shared pytest fixtures for tests/index/.

Fixtures
--------
fake_pil_image      – a tiny 32×32 RGB PIL image (no disk I/O).
fake_pil_image_rect – a non-square 48×32 RGB PIL image for aspect-ratio tests.
make_image_files    – factory that writes N tiny PNGs into a tmp dir and returns
                      their paths.
mock_siglip_model   – a SigLIPModel-shaped object with batch_size, embedding_dim
                      and embed_images mocked.  No HuggingFace model is loaded.
mock_sam_vit_model  – SamViTModel-shaped mock with a centred mask covering the
                      middle quarter of a 32×32 image.
extractor           – ObjectExtractor built with the mock SAM (gray background).
tmp_db_path         – temp directory for an isolated LanceDB store.
indexer             – fully constructed Indexer wired to the mock model + tmp DB.
tmp_indexing_config – temp config.yaml containing both `models:` and
                      `indexing:` sections, pointing at the tmp DB.
tmp_preprocess_config – temp config.yaml with models.sam_vit and preprocess
                        sections; used to test ObjectExtractor.from_config.

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


IMG_H, IMG_W = 32, 32


@pytest.fixture
def fake_pil_image() -> Image.Image:
    """Tiny 32×32 solid-colour RGB PIL image."""
    return Image.new("RGB", (IMG_W, IMG_H), color=(100, 150, 200))


@pytest.fixture
def fake_pil_image_rect() -> Image.Image:
    """Non-square 48×32 RGB PIL image (wider than tall)."""
    return Image.new("RGB", (48, 32), color=(200, 100, 50))


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


# ---------------------------------------------------------------------------
# Mock SAM ViT (for ObjectExtractor tests)
# ---------------------------------------------------------------------------


def _centered_mask(h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    """Bool array with a centred rectangle covering the middle quarter."""
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    return mask


def make_sam_vit_mock(
    masks: list | None = None,
    scores: list | None = None,
    boxes: list | None = None,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
) -> MagicMock:
    """Return a SamViTModel-shaped MagicMock with configurable invoke return value."""
    if masks is None:
        masks = [_centered_mask(img_h, img_w)]
    if scores is None:
        scores = [0.9] * len(masks)
    if boxes is None:
        boxes = [[img_w // 4, img_h // 4, 3 * img_w // 4, 3 * img_h // 4]] * len(masks)
    sam = MagicMock(name="SamViTModel")
    sam.invoke.return_value = {"masks": masks, "scores": scores, "boxes": boxes}
    return sam


@pytest.fixture
def mock_sam_vit_model() -> MagicMock:
    """SamViTModel mock that returns a single centred 32×32 mask."""
    return make_sam_vit_mock()


@pytest.fixture
def extractor(mock_sam_vit_model):
    """ObjectExtractor with gray background, top_n=1, default settings, mock SAM."""
    from src.index.preprocess import ObjectExtractor

    return ObjectExtractor(sam=mock_sam_vit_model, background="gray", output_size=224, margin=0.1, top_n=1)


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


@pytest.fixture
def tmp_preprocess_config(tmp_path, monkeypatch) -> Path:
    """A config.yaml with models.sam_vit and preprocess sections.

    Monkeypatches SamViTModel.from_config so no HF download occurs.
    """
    cfg = {
        "models": {
            "sam_vit": {
                "model_id": "facebook/sam-vit-base",
                "device": "auto",
                "points_per_side": 32,
                "points_per_batch": 64,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
            }
        },
        "preprocess": {
            "background": "black",
            "selection": "area_centrality",
            "margin": 0.05,
            "output_size": 64,
            "top_n": 2,
            "noise_mean": 100.0,
            "noise_std": 20.0,
        },
    }
    f = tmp_path / "config.yaml"
    f.write_text(yaml.dump(cfg))
    monkeypatch.setattr(
        "src.index.preprocess.SamViTModel.from_config",
        lambda path: make_sam_vit_mock(),
    )
    return f
