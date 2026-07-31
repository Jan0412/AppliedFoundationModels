"""Tests for src.models.clip.CLIPEmbedModel.

Mirrors tests/models/test_siglib.py — the two wrappers implement one contract,
so the contract is asserted the same way on both. The CLIP-specific parts are
the projection dim, the ``get_*_features`` entry points and text truncation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.models.clip import CLIPEmbedModel

from .conftest import CLIP_PROJECTION_DIM, CLIP_VISION_HIDDEN

MODEL_ID = "openai/clip-vit-base-patch16"


# ---------------------------------------------------------------------------
# Convenience fixture: a fully-constructed CLIPEmbedModel with mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def clip(mock_clip_patches):  # noqa: ARG001 – patches applied via fixture
    """Construct a CLIPEmbedModel with HF calls mocked out."""
    return CLIPEmbedModel(model_id=MODEL_ID, device="auto", batch_size=64)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_calls_automodel_from_pretrained_with_device_map(mock_clip_patches):
    """The model is loaded once, with device forwarded as device_map."""
    CLIPEmbedModel(model_id=MODEL_ID, device="auto")
    mock_clip_patches["AutoModel"].from_pretrained.assert_called_once_with(
        MODEL_ID, device_map="auto"
    )


def test_init_calls_autoprocessor_from_pretrained(mock_clip_patches):
    """The processor is loaded from the same model id."""
    CLIPEmbedModel(model_id=MODEL_ID, device="cpu")
    mock_clip_patches["AutoProcessor"].from_pretrained.assert_called_once_with(MODEL_ID)


def test_init_puts_model_in_eval_mode(mock_clip_patches):
    """Inference-only: the model is switched to eval at construction."""
    CLIPEmbedModel(model_id=MODEL_ID, device="auto")
    mock_clip_patches["model"].eval.assert_called_once()


def test_init_reads_embedding_dim_from_projection_dim(clip):
    """embedding_dim is the projection dim, not an encoder hidden size.

    Image and text meet only after their projection heads, so the projected
    width is what gets stored in LanceDB.
    """
    assert clip.embedding_dim == CLIP_PROJECTION_DIM


def test_init_does_not_use_the_vision_hidden_size(clip):
    """Guards the dim above against silently picking the vision tower's width."""
    assert clip.embedding_dim != CLIP_VISION_HIDDEN


def test_init_stores_batch_size(mock_clip_patches):
    """batch_size is stored for callers that batch before embed_images."""
    c = CLIPEmbedModel(model_id=MODEL_ID, device="auto", batch_size=32)
    assert c.batch_size == 32


def test_init_stores_device_string(mock_clip_patches):
    """The device string is kept for callers that need to know it."""
    c = CLIPEmbedModel(model_id=MODEL_ID, device="cuda")
    assert c.device == "cuda"


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------


def test_embed_text_calls_processor_with_text_list(clip, mock_clip_patches):
    """The query is passed to the processor as a single-element list."""
    clip.embed_text("a cat on a sofa")
    call_kwargs = mock_clip_patches["processor"].call_args.kwargs
    assert call_kwargs["text"] == ["a cat on a sofa"]


def test_embed_text_truncates_at_the_context_limit(clip, mock_clip_patches):
    """CLIP caps text at 77 tokens; long queries truncate instead of raising."""
    clip.embed_text("a " * 500)
    call_kwargs = mock_clip_patches["processor"].call_args.kwargs
    assert call_kwargs["truncation"] is True


def test_embed_text_does_not_pad_to_max_length(clip, mock_clip_patches):
    """Unlike SigLIP, CLIP pads to the longest sequence in the batch."""
    clip.embed_text("hello")
    call_kwargs = mock_clip_patches["processor"].call_args.kwargs
    assert call_kwargs["padding"] is True


def test_embed_text_goes_through_the_text_projection(clip, mock_clip_patches):
    """get_text_features applies text_projection; the raw text_model does not."""
    clip.embed_text("hello")
    assert mock_clip_patches["model"].get_text_features.called


def test_embed_text_returns_1d_ndarray(clip):
    """A single string yields a 1-D vector of embedding_dim."""
    vec = clip.embed_text("hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape == (CLIP_PROJECTION_DIM,)


def test_embed_text_returns_normalized_vector(clip):
    """Cosine similarity is a dot product downstream, so vectors are L2-normalised."""
    vec = clip.embed_text("normalisation check")
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# embed_images
# ---------------------------------------------------------------------------


def test_embed_images_calls_processor_with_images_kwarg(clip, mock_clip_patches, fake_pil_image):
    """Images are passed through the processor's images kwarg."""
    clip.embed_images([fake_pil_image])
    call_kwargs = mock_clip_patches["processor"].call_args.kwargs
    assert "images" in call_kwargs


def test_embed_images_does_not_pass_padding(clip, mock_clip_patches, fake_pil_image):
    """Padding is a text-side concern only."""
    clip.embed_images([fake_pil_image])
    call_kwargs = mock_clip_patches["processor"].call_args.kwargs
    assert "padding" not in call_kwargs


def test_embed_images_goes_through_the_visual_projection(clip, mock_clip_patches, fake_pil_image):
    """get_image_features applies visual_projection into the shared space."""
    clip.embed_images([fake_pil_image])
    assert mock_clip_patches["model"].get_image_features.called


def test_embed_images_returns_2d_ndarray(clip, fake_pil_image):
    """A list of images yields an (N, D) array."""
    vecs = clip.embed_images([fake_pil_image])
    assert isinstance(vecs, np.ndarray)
    assert vecs.ndim == 2
    assert vecs.shape == (1, CLIP_PROJECTION_DIM)


def test_embed_images_batch_dimension_matches_input(clip, fake_pil_image):
    """Row count follows the number of images given."""
    vecs = clip.embed_images([fake_pil_image, fake_pil_image])
    assert vecs.shape[0] == 2


def test_embed_images_returns_normalized_rows(clip, fake_pil_image):
    """Every row is L2-normalised, as the indexer assumes."""
    vecs = clip.embed_images([fake_pil_image])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# invoke — LCEL entry point
# ---------------------------------------------------------------------------


def test_invoke_returns_dict(clip):
    """invoke returns the LCEL-friendly dict payload."""
    result = clip.invoke("any string")
    assert isinstance(result, dict)


def test_invoke_dict_contains_embedding_key(clip):
    """The payload key is 'embedding' — same as SigLIP's."""
    result = clip.invoke("any string")
    assert "embedding" in result


def test_invoke_str_dispatches_to_embed_text(clip):
    """A str input routes to embed_text."""
    clip.embed_text = MagicMock(return_value=np.ones(CLIP_PROJECTION_DIM, dtype=np.float32))
    result = clip.invoke("a query string")
    clip.embed_text.assert_called_once_with("a query string")
    assert result["embedding"].shape == (CLIP_PROJECTION_DIM,)


def test_invoke_pil_image_dispatches_to_embed_images_and_unwraps(clip, fake_pil_image):
    """A single image routes to embed_images and is unwrapped to 1-D."""
    batch = np.ones((1, CLIP_PROJECTION_DIM), dtype=np.float32)
    clip.embed_images = MagicMock(return_value=batch)
    result = clip.invoke(fake_pil_image)
    clip.embed_images.assert_called_once_with([fake_pil_image])
    assert result["embedding"].ndim == 1


def test_invoke_list_of_pil_dispatches_to_embed_images(clip, fake_pil_image):
    """A list of images routes to embed_images and stays 2-D."""
    batch = np.ones((2, CLIP_PROJECTION_DIM), dtype=np.float32)
    clip.embed_images = MagicMock(return_value=batch)
    images = [fake_pil_image, fake_pil_image]
    result = clip.invoke(images)
    clip.embed_images.assert_called_once_with(images)
    assert result["embedding"].ndim == 2


def test_invoke_raises_typeerror_for_integer_input(clip):
    """Unsupported input types fail loudly rather than silently."""
    with pytest.raises(TypeError):
        clip.invoke(42)


def test_invoke_raises_typeerror_for_string_list(clip):
    """A list of strings is not a batch of images."""
    with pytest.raises(TypeError):
        clip.invoke(["not an image"])


def test_invoke_raises_typeerror_for_none(clip):
    """None is not embeddable."""
    with pytest.raises(TypeError):
        clip.invoke(None)


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_reads_the_clip_section(mock_clip_patches, tmp_config):
    """from_config uses models.clip — not models.siglip."""
    CLIPEmbedModel.from_config(tmp_config)
    mock_clip_patches["AutoModel"].from_pretrained.assert_called_once_with(
        MODEL_ID, device_map="auto"
    )


def test_from_config_missing_section_raises(tmp_path, mock_clip_patches):
    """A config without a models.clip section is a KeyError, not a default."""
    import yaml

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"models": {"siglip": {"model_id": "x"}}}))
    with pytest.raises(KeyError):
        CLIPEmbedModel.from_config(config_file)


# ---------------------------------------------------------------------------
# Interop with SigLIP — the shared contract
# ---------------------------------------------------------------------------


def test_shares_the_embedder_contract_with_siglip():
    """Both wrappers expose the attributes Indexer and EmbedQuery rely on."""
    from src.models.siglib import SigLIPModel

    for attr in ("embed_text", "embed_images", "invoke", "from_config"):
        assert hasattr(CLIPEmbedModel, attr), attr
        assert hasattr(SigLIPModel, attr), attr


def test_invoke_accepts_a_pil_image_subclass_instance(clip):
    """The dispatch is isinstance-based, so PIL subclasses still work."""
    img = Image.new("RGB", (8, 8))
    assert "embedding" in clip.invoke(img)
