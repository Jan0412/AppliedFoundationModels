"""Tests for src.models.grounding_dino.GroundingDINOModel."""
from __future__ import annotations

import pytest
import torch

from src.models.grounding_dino import GroundingDINOModel


MODEL_ID = "IDEA-Research/grounding-dino-base"


# ---------------------------------------------------------------------------
# Convenience fixture: a fully-constructed GroundingDINOModel with mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def dino(mock_dino_patches):  # noqa: ARG001 – patches applied via fixture
    """Construct a GroundingDINOModel with HF calls mocked out."""
    return GroundingDINOModel(
        model_id=MODEL_ID,
        device="auto",
        box_threshold=0.35,
        text_threshold=0.25,
    )


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_init_calls_auto_model_from_pretrained_with_device_map(mock_dino_patches):
    """__init__ passes device_map=device to AutoModelForZeroShotObjectDetection."""
    GroundingDINOModel(model_id=MODEL_ID, device="auto")
    mock_dino_patches["AutoModel"].from_pretrained.assert_called_once_with(
        MODEL_ID, device_map="auto"
    )


def test_init_calls_auto_processor_from_pretrained(mock_dino_patches):
    """__init__ calls AutoProcessor.from_pretrained with model_id."""
    GroundingDINOModel(model_id=MODEL_ID, device="cpu")
    mock_dino_patches["AutoProcessor"].from_pretrained.assert_called_once_with(MODEL_ID)


def test_init_puts_model_in_eval_mode(mock_dino_patches):
    """__init__ calls .eval() on the loaded model."""
    GroundingDINOModel(model_id=MODEL_ID, device="auto")
    mock_dino_patches["model"].eval.assert_called_once()


def test_init_stores_box_threshold(mock_dino_patches):  # noqa: ARG001
    """box_threshold is stored from the constructor argument."""
    dino = GroundingDINOModel(model_id=MODEL_ID, device="auto", box_threshold=0.7)
    assert dino.box_threshold == 0.7


def test_init_stores_text_threshold(mock_dino_patches):  # noqa: ARG001
    """text_threshold is stored from the constructor argument."""
    dino = GroundingDINOModel(model_id=MODEL_ID, device="auto", text_threshold=0.4)
    assert dino.text_threshold == 0.4


def test_init_stores_device_string(mock_dino_patches):  # noqa: ARG001
    """device string is stored as-is on the instance."""
    dino = GroundingDINOModel(model_id=MODEL_ID, device="cuda")
    assert dino.device == "cuda"


def test_init_uses_default_thresholds(mock_dino_patches):  # noqa: ARG001
    """Defaults match the published Grounding DINO recommendation (0.35 / 0.25)."""
    dino = GroundingDINOModel(model_id=MODEL_ID)
    assert dino.box_threshold == 0.35
    assert dino.text_threshold == 0.25


# ---------------------------------------------------------------------------
# invoke tests — processor inputs
# ---------------------------------------------------------------------------


def test_invoke_calls_processor_with_image(dino, mock_dino_patches, fake_pil_image):
    """invoke passes the image as images= to the DINO processor."""
    dino.invoke({"image": fake_pil_image, "text": "a cat ."})
    call_kwargs = mock_dino_patches["processor"].call_args.kwargs
    assert call_kwargs.get("images") is fake_pil_image


def test_invoke_passes_text_ending_with_period_unchanged(
    dino, mock_dino_patches, fake_pil_image
):
    """A prompt already ending with '.' is forwarded verbatim — no double period."""
    dino.invoke({"image": fake_pil_image, "text": "a cat ."})
    call_kwargs = mock_dino_patches["processor"].call_args.kwargs
    assert call_kwargs.get("text") == "a cat ."


def test_invoke_appends_period_when_missing(dino, mock_dino_patches, fake_pil_image):
    """A prompt without a trailing period gets ' .' appended (DINO requirement)."""
    dino.invoke({"image": fake_pil_image, "text": "a cat"})
    call_kwargs = mock_dino_patches["processor"].call_args.kwargs
    forwarded = call_kwargs.get("text")
    assert forwarded.rstrip().endswith(".")
    # Original text is preserved up to the appended terminator.
    assert "a cat" in forwarded


def test_invoke_appends_period_after_trailing_whitespace(
    dino, mock_dino_patches, fake_pil_image
):
    """Trailing whitespace before the period check is handled (rstrip + ' .')."""
    dino.invoke({"image": fake_pil_image, "text": "a cat   "})
    forwarded = mock_dino_patches["processor"].call_args.kwargs.get("text")
    assert forwarded.rstrip().endswith(".")


def test_invoke_calls_processor_with_pt_tensors(dino, mock_dino_patches, fake_pil_image):
    """invoke requests PyTorch tensors from the processor."""
    dino.invoke({"image": fake_pil_image, "text": "a chair ."})
    call_kwargs = mock_dino_patches["processor"].call_args.kwargs
    assert call_kwargs.get("return_tensors") == "pt"


# ---------------------------------------------------------------------------
# invoke tests — post-process arguments
# ---------------------------------------------------------------------------


def test_invoke_calls_post_process_with_stored_box_threshold_as_threshold(
    dino, mock_dino_patches, fake_pil_image
):
    """invoke maps stored box_threshold to the post-process ``threshold`` kwarg.

    The transformers API uses ``threshold=`` (not ``box_threshold=``) — this is
    the bug regression test for the rename in
    GroundingDinoProcessor.post_process_grounded_object_detection.
    """
    dino.invoke({"image": fake_pil_image, "text": "a chair ."})
    call_kwargs = (
        mock_dino_patches["processor"]
        .post_process_grounded_object_detection.call_args.kwargs
    )
    assert call_kwargs.get("threshold") == 0.35
    assert "box_threshold" not in call_kwargs


def test_invoke_calls_post_process_with_stored_text_threshold(
    dino, mock_dino_patches, fake_pil_image
):
    """invoke passes the stored text_threshold to post-processing."""
    dino.invoke({"image": fake_pil_image, "text": "a chair ."})
    call_kwargs = (
        mock_dino_patches["processor"]
        .post_process_grounded_object_detection.call_args.kwargs
    )
    assert call_kwargs.get("text_threshold") == 0.25


def test_invoke_passes_input_ids_positionally_to_post_process(
    dino, mock_dino_patches, fake_pil_image
):
    """invoke forwards the processor's input_ids tensor as the 2nd positional arg."""
    dino.invoke({"image": fake_pil_image, "text": "a chair ."})
    args, _ = mock_dino_patches[
        "processor"
    ].post_process_grounded_object_detection.call_args
    # Positional args: (model_output, input_ids)
    assert len(args) >= 2
    assert isinstance(args[1], torch.Tensor)
    assert args[1].dtype == torch.long


def test_invoke_calls_post_process_with_height_width_target_size(
    dino, mock_dino_patches, fake_pil_image
):
    """target_sizes is [(height, width)] derived from PIL.Image.size[::-1]."""
    dino.invoke({"image": fake_pil_image, "text": "a chair ."})
    call_kwargs = (
        mock_dino_patches["processor"]
        .post_process_grounded_object_detection.call_args.kwargs
    )
    target_sizes = call_kwargs.get("target_sizes")
    assert target_sizes == [fake_pil_image.size[::-1]]


# ---------------------------------------------------------------------------
# invoke tests — return value
# ---------------------------------------------------------------------------


def test_invoke_returns_dict(dino, fake_pil_image):
    """invoke returns a dict (the first element of the post-process list)."""
    result = dino.invoke({"image": fake_pil_image, "text": "laptop ."})
    assert isinstance(result, dict)


def test_invoke_result_has_boxes_key(dino, fake_pil_image):
    """invoke result contains 'boxes'."""
    result = dino.invoke({"image": fake_pil_image, "text": "laptop ."})
    assert "boxes" in result


def test_invoke_result_has_scores_key(dino, fake_pil_image):
    """invoke result contains 'scores'."""
    result = dino.invoke({"image": fake_pil_image, "text": "laptop ."})
    assert "scores" in result


def test_invoke_result_has_labels_key(dino, fake_pil_image):
    """invoke result contains 'labels'."""
    result = dino.invoke({"image": fake_pil_image, "text": "laptop ."})
    assert "labels" in result


def test_invoke_returns_first_element_of_postprocess_list(
    dino, mock_dino_patches, fake_pil_image
):
    """invoke returns post_process[0] — additional dicts in the list are ignored."""
    mock_dino_patches[
        "processor"
    ].post_process_grounded_object_detection.return_value = [
        {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
         "scores": torch.tensor([0.42]),
         "labels": ["expected"]},
        {"boxes": torch.tensor([[9.0, 9.0, 9.0, 9.0]]),
         "scores": torch.tensor([0.0]),
         "labels": ["should-not-appear"]},
    ]
    result = dino.invoke({"image": fake_pil_image, "text": "laptop ."})
    assert result["labels"] == ["expected"]


# ---------------------------------------------------------------------------
# invoke tests — error paths
# ---------------------------------------------------------------------------


def test_invoke_missing_image_key_raises_keyerror(dino):
    """invoke raises KeyError when 'image' is absent from the input dict."""
    with pytest.raises(KeyError, match="image"):
        dino.invoke({"text": "no image here ."})


def test_invoke_missing_text_key_raises_keyerror(dino, fake_pil_image):
    """invoke raises KeyError when 'text' is absent from the input dict."""
    with pytest.raises(KeyError, match="text"):
        dino.invoke({"image": fake_pil_image})


# ---------------------------------------------------------------------------
# from_config test
# ---------------------------------------------------------------------------


def test_from_config_constructs_with_yaml_values(mock_dino_patches, tmp_config):
    """from_config reads YAML and passes kwargs to __init__."""
    dino = GroundingDINOModel.from_config(tmp_config)
    mock_dino_patches["AutoModel"].from_pretrained.assert_called_once_with(
        "IDEA-Research/grounding-dino-base", device_map="auto"
    )
    assert dino.box_threshold == 0.35
    assert dino.text_threshold == 0.25


# ---------------------------------------------------------------------------
# LCEL chain composition test
# ---------------------------------------------------------------------------


def test_pipe_operator_returns_runnable_sequence(mock_dino_patches):  # noqa: ARG001
    """GroundingDINOModel supports | for LCEL chain composition."""
    from langchain_core.runnables import RunnableLambda, RunnableSequence

    dino = GroundingDINOModel(model_id=MODEL_ID, device="auto")
    chain = dino | RunnableLambda(lambda v: v.get("labels", []))
    assert isinstance(chain, RunnableSequence)
