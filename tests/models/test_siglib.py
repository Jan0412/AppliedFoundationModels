"""Tests for src.models.siglib.SigLIPModel."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.models.siglib import SigLIPModel


MODEL_ID = "google/siglip2-base-patch16-224"


# ---------------------------------------------------------------------------
# Convenience fixture: a fully-constructed SigLIPModel with mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def siglip(mock_siglip_patches):  # noqa: ARG001 – patches applied via fixture
    """Construct a SigLIPModel with HF calls mocked out."""
    return SigLIPModel(model_id=MODEL_ID, device="auto", batch_size=64)


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_init_calls_automodel_from_pretrained_with_device_map(mock_siglip_patches):
    """__init__ passes device_map=device to AutoModel.from_pretrained."""
    SigLIPModel(model_id=MODEL_ID, device="auto")
    mock_siglip_patches["AutoModel"].from_pretrained.assert_called_once_with(
        MODEL_ID, device_map="auto"
    )


def test_init_calls_autoprocessor_from_pretrained(mock_siglip_patches):
    """__init__ calls AutoProcessor.from_pretrained with the model_id."""
    SigLIPModel(model_id=MODEL_ID, device="cpu")
    mock_siglip_patches["AutoProcessor"].from_pretrained.assert_called_once_with(MODEL_ID)


def test_init_puts_model_in_eval_mode(mock_siglip_patches):
    """__init__ calls .eval() on the loaded model."""
    SigLIPModel(model_id=MODEL_ID, device="auto")
    mock_siglip_patches["model"].eval.assert_called_once()


def test_init_sets_embedding_dim_from_model_config(mock_siglip_patches):
    """embedding_dim is read from model.config.text_config.hidden_size (768)."""
    sig = SigLIPModel(model_id=MODEL_ID, device="auto")
    assert sig.embedding_dim == 768


def test_init_stores_batch_size(mock_siglip_patches):
    """batch_size is stored from the constructor argument."""
    sig = SigLIPModel(model_id=MODEL_ID, device="auto", batch_size=32)
    assert sig.batch_size == 32


def test_init_stores_device_string(mock_siglip_patches):
    """device string is stored as-is on the instance."""
    sig = SigLIPModel(model_id=MODEL_ID, device="cuda")
    assert sig.device == "cuda"


# ---------------------------------------------------------------------------
# embed_text tests
# ---------------------------------------------------------------------------


def test_embed_text_calls_processor_with_text_list(siglip, mock_siglip_patches):
    """embed_text wraps the input in a list and passes it as text= to the processor."""
    siglip.embed_text("a cat on a sofa")
    call_kwargs = mock_siglip_patches["processor"].call_args.kwargs
    assert call_kwargs.get("text") == ["a cat on a sofa"]


def test_embed_text_calls_processor_with_padding_max_length(siglip, mock_siglip_patches):
    """embed_text uses padding='max_length' (required by SigLIP text encoder)."""
    siglip.embed_text("hello")
    call_kwargs = mock_siglip_patches["processor"].call_args.kwargs
    assert call_kwargs.get("padding") == "max_length"
    assert call_kwargs.get("return_tensors") == "pt"


def test_embed_text_returns_1d_ndarray(siglip):
    """embed_text returns a 1-D numpy array."""
    vec = siglip.embed_text("hello world")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape == (768,)


def test_embed_text_returns_normalized_vector(siglip):
    """embed_text returns an L2-normalised vector (unit norm)."""
    vec = siglip.embed_text("normalisation check")
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# embed_images tests
# ---------------------------------------------------------------------------


def test_embed_images_calls_processor_with_images_kwarg(siglip, mock_siglip_patches, fake_pil_image):
    """embed_images passes the image list as images= to the processor."""
    siglip.embed_images([fake_pil_image])
    call_kwargs = mock_siglip_patches["processor"].call_args.kwargs
    assert "images" in call_kwargs


def test_embed_images_does_not_pass_padding(siglip, mock_siglip_patches, fake_pil_image):
    """embed_images does NOT pass padding= (unlike embed_text)."""
    siglip.embed_images([fake_pil_image])
    call_kwargs = mock_siglip_patches["processor"].call_args.kwargs
    assert "padding" not in call_kwargs


def test_embed_images_returns_2d_ndarray(siglip, fake_pil_image):
    """embed_images returns a 2-D numpy array of shape (N, embedding_dim)."""
    vecs = siglip.embed_images([fake_pil_image])
    assert isinstance(vecs, np.ndarray)
    assert vecs.ndim == 2
    assert vecs.shape == (1, 768)


def test_embed_images_batch_dimension_matches_input(siglip, fake_pil_image):
    """embed_images returns the correct batch dimension for multi-image input."""
    vecs = siglip.embed_images([fake_pil_image, fake_pil_image])
    assert vecs.shape == (2, 768)


def test_embed_images_returns_normalized_rows(siglip, fake_pil_image):
    """embed_images returns L2-normalised row vectors."""
    vecs = siglip.embed_images([fake_pil_image])
    norms = np.linalg.norm(vecs, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# invoke dispatch tests
# ---------------------------------------------------------------------------


def test_invoke_returns_dict(siglip):
    """invoke always returns a dict."""
    result = siglip.invoke("any string")
    assert isinstance(result, dict)


def test_invoke_dict_contains_embedding_key(siglip):
    """invoke result always contains the 'embedding' key."""
    result = siglip.invoke("any string")
    assert "embedding" in result


def test_invoke_str_dispatches_to_embed_text(siglip):
    """invoke(str) calls embed_text and wraps the result in {'embedding': ...}."""
    siglip.embed_text = MagicMock(return_value=np.ones(768, dtype=np.float32))
    result = siglip.invoke("a query string")
    siglip.embed_text.assert_called_once_with("a query string")
    assert isinstance(result["embedding"], np.ndarray)


def test_invoke_pil_image_dispatches_to_embed_images_and_unwraps(siglip, fake_pil_image):
    """invoke(PIL.Image) calls embed_images([img]) and embedding is 1-D."""
    batch = np.ones((1, 768), dtype=np.float32)
    siglip.embed_images = MagicMock(return_value=batch)

    result = siglip.invoke(fake_pil_image)

    siglip.embed_images.assert_called_once_with([fake_pil_image])
    assert isinstance(result, dict)
    assert result["embedding"].ndim == 1
    assert result["embedding"].shape == (768,)


def test_invoke_list_of_pil_dispatches_to_embed_images(siglip, fake_pil_image):
    """invoke(list[PIL.Image]) calls embed_images and embedding is 2-D."""
    batch = np.ones((2, 768), dtype=np.float32)
    siglip.embed_images = MagicMock(return_value=batch)

    images = [fake_pil_image, fake_pil_image]
    result = siglip.invoke(images)

    siglip.embed_images.assert_called_once_with(images)
    assert isinstance(result, dict)
    assert result["embedding"].shape == (2, 768)


def test_invoke_raises_typeerror_for_integer_input(siglip):
    """invoke raises TypeError for unsupported types (e.g. int)."""
    with pytest.raises(TypeError, match="unsupported input type"):
        siglip.invoke(42)


def test_invoke_raises_typeerror_for_string_list(siglip):
    """invoke raises TypeError for list[str] (not list[PIL.Image])."""
    with pytest.raises(TypeError):
        siglip.invoke(["not an image"])


def test_invoke_raises_typeerror_for_none(siglip):
    """invoke raises TypeError for None."""
    with pytest.raises(TypeError):
        siglip.invoke(None)


# ---------------------------------------------------------------------------
# from_config test
# ---------------------------------------------------------------------------


def test_from_config_constructs_with_yaml_values(mock_siglip_patches, tmp_config):
    """from_config reads YAML and passes kwargs to __init__."""
    sig = SigLIPModel.from_config(tmp_config)
    mock_siglip_patches["AutoModel"].from_pretrained.assert_called_once_with(
        "google/siglip2-base-patch16-224", device_map="auto"
    )
    assert sig.batch_size == 64
    assert sig.embedding_dim == 768


# ---------------------------------------------------------------------------
# LCEL chain composition test
# ---------------------------------------------------------------------------


def test_pipe_operator_returns_runnable_sequence(mock_siglip_patches):
    """SigLIPModel supports | for LCEL chain composition."""
    from langchain_core.runnables import RunnableLambda, RunnableSequence

    sig = SigLIPModel(model_id=MODEL_ID, device="auto")
    chain = sig | RunnableLambda(lambda v: v.shape)
    assert isinstance(chain, RunnableSequence)
