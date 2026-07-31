"""Shared pytest fixtures for tests/models/.

Fixtures
--------
tmp_config       – a temp config.yaml whose values match the real config.yaml.
fake_pil_image   – a tiny 32×32 RGB PIL.Image (no disk I/O).
mock_siglip_patches – monkeypatches AutoModel / AutoProcessor in src.models.siglib.
mock_clip_patches   – monkeypatches AutoModel / AutoProcessor in src.models.clip.
mock_sam_patches    – monkeypatches Sam3Model / Sam3Processor in src.models.sam.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import yaml
from PIL import Image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_config(tmp_path):
    """Write a minimal config.yaml to a temp directory and return its Path."""
    config = {
        "models": {
            "embedder": "siglip",
            "siglip": {
                "model_id": "google/siglip2-base-patch16-224",
                "device": "auto",
                "batch_size": 64,
            },
            "clip": {
                "model_id": "openai/clip-vit-base-patch16",
                "device": "auto",
                "batch_size": 64,
            },
            "sam": {
                "model_id": "facebook/sam3",
                "device": "auto",
                "threshold": 0.5,
                "mask_threshold": 0.5,
            },
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config))
    return config_file


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_pil_image():
    """A tiny 32×32 RGB PIL image — no disk I/O required."""
    return Image.new("RGB", (32, 32), color=(100, 150, 200))


# ---------------------------------------------------------------------------
# SigLIP mock helpers
# ---------------------------------------------------------------------------


class _MockBatchEncoding(dict):
    """Minimal dict subclass that supports the `.to(device)` call."""

    def to(self, device):  # noqa: ARG002
        return self


def _make_siglip_model_mock() -> MagicMock:
    """Return a MagicMock that behaves like a loaded SigLIP AutoModel."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.config.text_config.hidden_size = 768
    model.eval.return_value = model  # .eval() chains back to self

    def _vision_forward(**kwargs):
        """Return batch-size-aware pooler_output from pixel_values shape."""
        pixel_values = kwargs.get("pixel_values", torch.ones(1, 3, 224, 224))
        B = pixel_values.shape[0] if hasattr(pixel_values, "shape") else 1
        out = MagicMock()
        out.pooler_output = torch.ones(B, 768)
        return out

    model.vision_model.side_effect = _vision_forward

    def _text_forward(**kwargs):  # noqa: ARG001
        out = MagicMock()
        out.pooler_output = torch.ones(1, 768)
        return out

    model.text_model.side_effect = _text_forward
    return model


def _make_siglip_processor_mock() -> MagicMock:
    """Return a MagicMock that behaves like a SigLIP AutoProcessor."""
    processor = MagicMock()

    def _proc_call(*args, **kwargs):  # noqa: ARG001
        images = kwargs.get("images", None)
        B = len(images) if isinstance(images, (list, tuple)) else 1
        return _MockBatchEncoding(
            pixel_values=torch.ones(B, 3, 224, 224),
            input_ids=torch.zeros(B, 64, dtype=torch.long),
            attention_mask=torch.ones(B, 64, dtype=torch.long),
        )

    processor.side_effect = _proc_call
    return processor


@pytest.fixture
def mock_siglip_patches(monkeypatch):
    """Patch AutoModel and AutoProcessor in src.models.siglib.

    Returns a dict with keys ``model``, ``processor``, ``AutoModel``,
    ``AutoProcessor`` for inspection in tests.
    """
    mock_model = _make_siglip_model_mock()
    mock_processor = _make_siglip_processor_mock()

    mock_auto_model_cls = MagicMock()
    mock_auto_model_cls.from_pretrained.return_value = mock_model

    mock_auto_processor_cls = MagicMock()
    mock_auto_processor_cls.from_pretrained.return_value = mock_processor

    monkeypatch.setattr("src.models.siglib.AutoModel", mock_auto_model_cls)
    monkeypatch.setattr("src.models.siglib.AutoProcessor", mock_auto_processor_cls)

    return {
        "model": mock_model,
        "processor": mock_processor,
        "AutoModel": mock_auto_model_cls,
        "AutoProcessor": mock_auto_processor_cls,
    }


# ---------------------------------------------------------------------------
# CLIP mock helpers
# ---------------------------------------------------------------------------

#: CLIP ViT-B/16: 512-d projected space, 768-d vision hidden. The wrapper must
#: read the projection dim, so the two are kept deliberately different here.
CLIP_PROJECTION_DIM = 512
CLIP_VISION_HIDDEN = 768


def _make_clip_model_mock() -> MagicMock:
    """Return a MagicMock that behaves like a loaded CLIP AutoModel."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.config.projection_dim = CLIP_PROJECTION_DIM
    model.config.vision_config.hidden_size = CLIP_VISION_HIDDEN
    model.config.text_config.hidden_size = CLIP_PROJECTION_DIM
    model.eval.return_value = model  # .eval() chains back to self

    def _image_features(**kwargs):
        """Batch-size-aware projected pooler_output, from pixel_values shape."""
        pixel_values = kwargs.get("pixel_values", torch.ones(1, 3, 224, 224))
        B = pixel_values.shape[0] if hasattr(pixel_values, "shape") else 1
        out = MagicMock()
        out.pooler_output = torch.ones(B, CLIP_PROJECTION_DIM)
        return out

    model.get_image_features.side_effect = _image_features

    def _text_features(**kwargs):  # noqa: ARG001
        out = MagicMock()
        out.pooler_output = torch.ones(1, CLIP_PROJECTION_DIM)
        return out

    model.get_text_features.side_effect = _text_features
    return model


@pytest.fixture
def mock_clip_patches(monkeypatch):
    """Patch AutoModel and AutoProcessor in src.models.clip.

    Returns a dict with keys ``model``, ``processor``, ``AutoModel``,
    ``AutoProcessor`` for inspection in tests.
    """
    mock_model = _make_clip_model_mock()
    # The processor contract is the same as SigLIP's, so it is reused.
    mock_processor = _make_siglip_processor_mock()

    mock_auto_model_cls = MagicMock()
    mock_auto_model_cls.from_pretrained.return_value = mock_model

    mock_auto_processor_cls = MagicMock()
    mock_auto_processor_cls.from_pretrained.return_value = mock_processor

    monkeypatch.setattr("src.models.clip.AutoModel", mock_auto_model_cls)
    monkeypatch.setattr("src.models.clip.AutoProcessor", mock_auto_processor_cls)

    return {
        "model": mock_model,
        "processor": mock_processor,
        "AutoModel": mock_auto_model_cls,
        "AutoProcessor": mock_auto_processor_cls,
    }


# ---------------------------------------------------------------------------
# SAM mock helpers
# ---------------------------------------------------------------------------


def _make_sam_model_mock() -> MagicMock:
    """Return a MagicMock that behaves like a loaded SAM3 model."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.eval.return_value = model  # .eval() chains back to self
    # model(**inp) → model.return_value (a MagicMock) — default MagicMock behaviour
    return model


def _make_sam_processor_mock() -> MagicMock:
    """Return a MagicMock that behaves like a SAM3Processor."""
    processor = MagicMock()

    def _proc_call(*args, **kwargs):  # noqa: ARG001
        return _MockBatchEncoding(
            pixel_values=torch.ones(1, 3, 1024, 1024),
            input_ids=torch.zeros(1, 32, dtype=torch.long),
            original_sizes=torch.tensor([[480, 640]]),
        )

    processor.side_effect = _proc_call
    processor.post_process_instance_segmentation.return_value = [
        {
            "masks": [torch.zeros(480, 640, dtype=torch.bool)],
            "scores": torch.tensor([0.9]),
            "boxes": torch.tensor([[10.0, 20.0, 100.0, 200.0]]),
        }
    ]
    return processor


@pytest.fixture
def mock_sam_patches(monkeypatch):
    """Patch Sam3Model and Sam3Processor in src.models.sam.

    Returns a dict with keys ``model``, ``processor``, ``Sam3Model``,
    ``Sam3Processor`` for inspection in tests.
    """
    mock_model = _make_sam_model_mock()
    mock_processor = _make_sam_processor_mock()

    mock_sam3_model_cls = MagicMock()
    mock_sam3_model_cls.from_pretrained.return_value = mock_model

    mock_sam3_processor_cls = MagicMock()
    mock_sam3_processor_cls.from_pretrained.return_value = mock_processor

    monkeypatch.setattr("src.models.sam.Sam3Model", mock_sam3_model_cls)
    monkeypatch.setattr("src.models.sam.Sam3Processor", mock_sam3_processor_cls)

    return {
        "model": mock_model,
        "processor": mock_processor,
        "Sam3Model": mock_sam3_model_cls,
        "Sam3Processor": mock_sam3_processor_cls,
    }
