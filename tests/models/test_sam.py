"""Tests for src.models.sam.SAMModel."""
from __future__ import annotations

import pytest
import torch

from src.models.sam import SAMModel


MODEL_ID = "facebook/sam3"


# ---------------------------------------------------------------------------
# Convenience fixture: a fully-constructed SAMModel with mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def sam(mock_sam_patches):  # noqa: ARG001 – patches applied via fixture
    """Construct a SAMModel with HF calls mocked out."""
    return SAMModel(model_id=MODEL_ID, device="auto", threshold=0.5, mask_threshold=0.5)


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_init_calls_sam3model_from_pretrained_with_device_map(mock_sam_patches):
    """__init__ passes device_map=device to Sam3Model.from_pretrained."""
    SAMModel(model_id=MODEL_ID, device="auto")
    mock_sam_patches["Sam3Model"].from_pretrained.assert_called_once_with(
        MODEL_ID, device_map="auto"
    )


def test_init_calls_sam3processor_from_pretrained(mock_sam_patches):
    """__init__ calls Sam3Processor.from_pretrained with model_id."""
    SAMModel(model_id=MODEL_ID, device="cpu")
    mock_sam_patches["Sam3Processor"].from_pretrained.assert_called_once_with(MODEL_ID)


def test_init_puts_model_in_eval_mode(mock_sam_patches):
    """__init__ calls .eval() on the loaded model."""
    SAMModel(model_id=MODEL_ID, device="auto")
    mock_sam_patches["model"].eval.assert_called_once()


def test_init_stores_threshold(mock_sam_patches):
    """threshold is stored from the constructor argument."""
    sam = SAMModel(model_id=MODEL_ID, device="auto", threshold=0.7)
    assert sam.threshold == 0.7


def test_init_stores_mask_threshold(mock_sam_patches):
    """mask_threshold is stored from the constructor argument."""
    sam = SAMModel(model_id=MODEL_ID, device="auto", mask_threshold=0.3)
    assert sam.mask_threshold == 0.3


def test_init_stores_device_string(mock_sam_patches):
    """device string is stored as-is on the instance."""
    sam = SAMModel(model_id=MODEL_ID, device="cuda")
    assert sam.device == "cuda"


# ---------------------------------------------------------------------------
# invoke tests
# ---------------------------------------------------------------------------


def test_invoke_calls_processor_with_image(sam, mock_sam_patches, fake_pil_image):
    """invoke passes the image as images= to the SAM processor."""
    sam.invoke({"image": fake_pil_image, "text": "a cat"})
    call_kwargs = mock_sam_patches["processor"].call_args.kwargs
    assert call_kwargs.get("images") == fake_pil_image


def test_invoke_calls_processor_with_text(sam, mock_sam_patches, fake_pil_image):
    """invoke passes the text prompt as text= to the SAM processor."""
    sam.invoke({"image": fake_pil_image, "text": "a cat"})
    call_kwargs = mock_sam_patches["processor"].call_args.kwargs
    assert call_kwargs.get("text") == "a cat"


def test_invoke_calls_processor_with_pt_tensors(sam, mock_sam_patches, fake_pil_image):
    """invoke requests PyTorch tensors from the processor."""
    sam.invoke({"image": fake_pil_image, "text": "a chair"})
    call_kwargs = mock_sam_patches["processor"].call_args.kwargs
    assert call_kwargs.get("return_tensors") == "pt"


def test_invoke_calls_post_process_with_stored_threshold(sam, mock_sam_patches, fake_pil_image):
    """invoke passes the stored threshold to post_process_instance_segmentation."""
    sam.invoke({"image": fake_pil_image, "text": "a chair"})
    call_kwargs = (
        mock_sam_patches["processor"].post_process_instance_segmentation.call_args.kwargs
    )
    assert call_kwargs.get("threshold") == 0.5


def test_invoke_calls_post_process_with_stored_mask_threshold(
    sam, mock_sam_patches, fake_pil_image
):
    """invoke passes the stored mask_threshold to post_process_instance_segmentation."""
    sam.invoke({"image": fake_pil_image, "text": "a chair"})
    call_kwargs = (
        mock_sam_patches["processor"].post_process_instance_segmentation.call_args.kwargs
    )
    assert call_kwargs.get("mask_threshold") == 0.5


def test_invoke_returns_dict(sam, fake_pil_image):
    """invoke returns a dict."""
    result = sam.invoke({"image": fake_pil_image, "text": "laptop"})
    assert isinstance(result, dict)


def test_invoke_result_has_masks_key(sam, fake_pil_image):
    """invoke result contains 'masks'."""
    result = sam.invoke({"image": fake_pil_image, "text": "laptop"})
    assert "masks" in result


def test_invoke_result_has_scores_key(sam, fake_pil_image):
    """invoke result contains 'scores'."""
    result = sam.invoke({"image": fake_pil_image, "text": "laptop"})
    assert "scores" in result


def test_invoke_result_has_boxes_key(sam, fake_pil_image):
    """invoke result contains 'boxes'."""
    result = sam.invoke({"image": fake_pil_image, "text": "laptop"})
    assert "boxes" in result


def test_invoke_missing_image_key_raises_keyerror(sam):
    """invoke raises KeyError when 'image' is absent from the input dict."""
    with pytest.raises(KeyError, match="image"):
        sam.invoke({"text": "no image here"})


def test_invoke_missing_text_key_raises_keyerror(sam, fake_pil_image):
    """invoke raises KeyError when 'text' is absent from the input dict."""
    with pytest.raises(KeyError, match="text"):
        sam.invoke({"image": fake_pil_image})


# ---------------------------------------------------------------------------
# from_config test
# ---------------------------------------------------------------------------


def test_from_config_constructs_with_yaml_values(mock_sam_patches, tmp_config):
    """from_config reads YAML and passes kwargs to __init__."""
    sam = SAMModel.from_config(tmp_config)
    mock_sam_patches["Sam3Model"].from_pretrained.assert_called_once_with(
        "facebook/sam3", device_map="auto"
    )
    assert sam.threshold == 0.5
    assert sam.mask_threshold == 0.5


# ---------------------------------------------------------------------------
# LCEL chain composition test
# ---------------------------------------------------------------------------


def test_pipe_operator_returns_runnable_sequence(mock_sam_patches):
    """SAMModel supports | for LCEL chain composition."""
    from langchain_core.runnables import RunnableLambda, RunnableSequence

    sam = SAMModel(model_id=MODEL_ID, device="auto")
    chain = sam | RunnableLambda(lambda v: v.get("masks", []))
    assert isinstance(chain, RunnableSequence)
