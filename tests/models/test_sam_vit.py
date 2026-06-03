"""Tests for src.models.sam_vit.SamViTModel."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.sam_vit import SamViTModel, _to_bool_array


MODEL_ID = "facebook/sam-vit-base"


# ---------------------------------------------------------------------------
# Convenience fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sam_vit(mock_sam_vit_patches):  # noqa: ARG001
    """SamViTModel with all HF calls mocked out."""
    return SamViTModel(model_id=MODEL_ID, device="auto")


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_init_calls_sammodel_from_pretrained_with_device_map(mock_sam_vit_patches):
    """__init__ passes device_map=device to SamModel.from_pretrained."""
    SamViTModel(model_id=MODEL_ID, device="auto")
    mock_sam_vit_patches["SamModel"].from_pretrained.assert_called_once_with(
        MODEL_ID, device_map="auto"
    )


def test_init_calls_samprocessor_from_pretrained(mock_sam_vit_patches):
    """__init__ calls SamProcessor.from_pretrained with model_id."""
    SamViTModel(model_id=MODEL_ID, device="cpu")
    mock_sam_vit_patches["SamProcessor"].from_pretrained.assert_called_once_with(MODEL_ID)


def test_init_puts_model_in_eval_mode(mock_sam_vit_patches):
    """__init__ calls .eval() on the loaded model."""
    SamViTModel(model_id=MODEL_ID, device="auto")
    mock_sam_vit_patches["model"].eval.assert_called_once()


def test_init_builds_mask_generation_pipeline(mock_sam_vit_patches):
    """__init__ calls pipeline with task='mask-generation' and the loaded model."""
    SamViTModel(model_id=MODEL_ID, device="auto")
    mock_sam_vit_patches["pipeline_factory"].assert_called_once()
    call = mock_sam_vit_patches["pipeline_factory"].call_args
    # task may be passed positionally or as a keyword depending on call site
    task_from_args = call.args[0] if call.args else None
    task_from_kwargs = call.kwargs.get("task")
    assert task_from_args == "mask-generation" or task_from_kwargs == "mask-generation"


def test_init_stores_points_per_side(mock_sam_vit_patches):
    sam = SamViTModel(model_id=MODEL_ID, points_per_side=16)
    assert sam.points_per_side == 16


def test_init_stores_points_per_batch(mock_sam_vit_patches):
    sam = SamViTModel(model_id=MODEL_ID, points_per_batch=32)
    assert sam.points_per_batch == 32


def test_init_stores_pred_iou_thresh(mock_sam_vit_patches):
    sam = SamViTModel(model_id=MODEL_ID, pred_iou_thresh=0.75)
    assert sam.pred_iou_thresh == 0.75


def test_init_stores_stability_score_thresh(mock_sam_vit_patches):
    sam = SamViTModel(model_id=MODEL_ID, stability_score_thresh=0.80)
    assert sam.stability_score_thresh == 0.80


def test_init_stores_device_string(mock_sam_vit_patches):
    sam = SamViTModel(model_id=MODEL_ID, device="cuda")
    assert sam.device == "cuda"


# ---------------------------------------------------------------------------
# invoke tests
# ---------------------------------------------------------------------------


def test_invoke_calls_pipeline_with_image(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke passes the PIL image to the mask_generator pipeline."""
    sam_vit.invoke(fake_pil_image)
    mock_sam_vit_patches["pipeline"].assert_called_once()
    call_args = mock_sam_vit_patches["pipeline"].call_args
    assert call_args.args[0] is fake_pil_image


def test_invoke_forwards_points_per_side(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke passes points_per_side as points_per_crop (transformers 5.x kwarg name)."""
    sam_vit.invoke(fake_pil_image)
    call_kwargs = mock_sam_vit_patches["pipeline"].call_args.kwargs
    assert call_kwargs.get("points_per_crop") == sam_vit.points_per_side


def test_invoke_forwards_pred_iou_thresh(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke passes pred_iou_thresh to the pipeline call."""
    sam_vit.invoke(fake_pil_image)
    call_kwargs = mock_sam_vit_patches["pipeline"].call_args.kwargs
    assert call_kwargs.get("pred_iou_thresh") == sam_vit.pred_iou_thresh


def test_invoke_returns_dict(sam_vit, fake_pil_image):
    """invoke returns a dict."""
    result = sam_vit.invoke(fake_pil_image)
    assert isinstance(result, dict)


def test_invoke_result_has_masks_key(sam_vit, fake_pil_image):
    result = sam_vit.invoke(fake_pil_image)
    assert "masks" in result


def test_invoke_result_has_scores_key(sam_vit, fake_pil_image):
    result = sam_vit.invoke(fake_pil_image)
    assert "scores" in result


def test_invoke_result_has_boxes_key(sam_vit, fake_pil_image):
    result = sam_vit.invoke(fake_pil_image)
    assert "boxes" in result


def test_invoke_masks_are_bool_numpy_arrays(sam_vit, fake_pil_image):
    """invoke converts pipeline masks to bool numpy arrays."""
    result = sam_vit.invoke(fake_pil_image)
    assert len(result["masks"]) > 0
    for m in result["masks"]:
        assert isinstance(m, np.ndarray)
        assert m.dtype == bool


def test_invoke_scores_are_floats(sam_vit, fake_pil_image):
    """invoke returns scores as a list of floats."""
    result = sam_vit.invoke(fake_pil_image)
    for s in result["scores"]:
        assert isinstance(s, float)


def test_invoke_boxes_are_xyxy(sam_vit, fake_pil_image):
    """invoke converts [x,y,w,h] bbox to [x0,y0,x1,y1] format."""
    result = sam_vit.invoke(fake_pil_image)
    for box in result["boxes"]:
        assert len(box) == 4
        x0, y0, x1, y1 = box
        assert x1 >= x0
        assert y1 >= y0


def test_invoke_non_image_raises_typeerror(sam_vit):
    """invoke raises TypeError for non-PIL input."""
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        sam_vit.invoke("not an image")


def test_invoke_non_image_dict_raises_typeerror(sam_vit):
    """invoke raises TypeError for a dict input (unlike SAMModel which needs one)."""
    with pytest.raises(TypeError):
        sam_vit.invoke({"image": Image.new("RGB", (32, 32))})


def test_invoke_forwards_points_per_batch(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke passes points_per_batch to the pipeline call."""
    sam_vit.invoke(fake_pil_image)
    call_kwargs = mock_sam_vit_patches["pipeline"].call_args.kwargs
    assert call_kwargs.get("points_per_batch") == sam_vit.points_per_batch


def test_invoke_forwards_stability_score_thresh(
    sam_vit, mock_sam_vit_patches, fake_pil_image
):
    """invoke passes stability_score_thresh to the pipeline call."""
    sam_vit.invoke(fake_pil_image)
    call_kwargs = mock_sam_vit_patches["pipeline"].call_args.kwargs
    assert call_kwargs.get("stability_score_thresh") == sam_vit.stability_score_thresh


def test_invoke_box_passthrough_is_exact(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke passes XYXY bounding boxes from the pipeline dict through unchanged."""
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [np.ones((4, 4), dtype=bool)],
        "scores": torch.tensor([0.5]),
        "bounding_boxes": torch.tensor([[10.0, 20.0, 40.0, 60.0]]),
    }
    result = sam_vit.invoke(fake_pil_image)
    assert result["boxes"] == [[10.0, 20.0, 40.0, 60.0]]


def test_invoke_box_values_are_floats(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke returns box coordinates as Python floats."""
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [np.ones((2, 2), dtype=bool)],
        "scores": torch.tensor([0.9]),
        "bounding_boxes": torch.tensor([[1.0, 2.0, 4.0, 6.0]]),
    }
    result = sam_vit.invoke(fake_pil_image)
    for coord in result["boxes"][0]:
        assert isinstance(coord, float)


def test_invoke_uses_masks_from_dict_output(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke reads masks from the 'masks' key of the pipeline dict."""
    mask = np.ones((3, 3), dtype=bool)
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [mask],
        "scores": torch.tensor([0.5]),
        "bounding_boxes": torch.tensor([[0.0, 0.0, 3.0, 3.0]]),
    }
    result = sam_vit.invoke(fake_pil_image)
    assert result["masks"][0].shape == (3, 3)
    assert result["masks"][0].all()


def test_invoke_reads_scores_from_dict_output(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke reads scores from the 'scores' tensor in the pipeline dict."""
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [np.ones((2, 2), dtype=bool)],
        "scores": torch.tensor([0.7]),
        "bounding_boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
    }
    result = sam_vit.invoke(fake_pil_image)
    assert result["scores"] == [pytest.approx(0.7)]


def test_invoke_bbox_derived_from_mask_when_boxes_absent(
    sam_vit, mock_sam_vit_patches, fake_pil_image
):
    """When 'bounding_boxes' is absent, invoke derives XYXY bbox from mask pixels."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True  # 2×2 block at (1,1)–(2,2)
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [mask],
        "scores": torch.tensor([0.5]),
        # no "bounding_boxes" key
    }
    result = sam_vit.invoke(fake_pil_image)
    assert result["boxes"] == [[1.0, 1.0, 2.0, 2.0]]


def test_invoke_preserves_scores_from_dict(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke reports the pipeline score unchanged."""
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [np.ones((2, 2), dtype=bool)],
        "scores": torch.tensor([0.42]),
        "bounding_boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
    }
    result = sam_vit.invoke(fake_pil_image)
    assert result["scores"] == [pytest.approx(0.42)]


def test_invoke_handles_multiple_masks(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke returns one entry per pipeline detection, in order."""
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [np.ones((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)],
        "scores": torch.tensor([0.9, 0.6]),
        "bounding_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 7.0, 7.0]]),
    }
    result = sam_vit.invoke(fake_pil_image)
    assert len(result["masks"]) == 2
    assert result["scores"] == [pytest.approx(0.9), pytest.approx(0.6)]
    assert result["boxes"] == [[0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 7.0, 7.0]]


def test_invoke_empty_output_returns_empty_lists(
    sam_vit, mock_sam_vit_patches, fake_pil_image
):
    """invoke returns empty lists when the pipeline detects nothing."""
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [],
        "scores": torch.tensor([]),
        "bounding_boxes": torch.zeros(0, 4),
    }
    result = sam_vit.invoke(fake_pil_image)
    assert result == {"masks": [], "scores": [], "boxes": []}


def test_invoke_result_lists_have_equal_length(sam_vit, fake_pil_image):
    """masks, scores and boxes are always parallel lists."""
    result = sam_vit.invoke(fake_pil_image)
    assert len(result["masks"]) == len(result["scores"]) == len(result["boxes"])


def test_invoke_preserves_mask_content(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke returns the exact mask the pipeline produced (no mangling)."""
    mask = np.array([[True, False], [False, True]])
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [mask],
        "scores": torch.tensor([0.5]),
        "bounding_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
    }
    result = sam_vit.invoke(fake_pil_image)
    np.testing.assert_array_equal(result["masks"][0], mask)


def test_invoke_forwards_output_bboxes_mask(sam_vit, mock_sam_vit_patches, fake_pil_image):
    """invoke requests bounding boxes from the pipeline (output_bboxes_mask=True)."""
    sam_vit.invoke(fake_pil_image)
    call_kwargs = mock_sam_vit_patches["pipeline"].call_args.kwargs
    assert call_kwargs.get("output_bboxes_mask") is True


def test_invoke_box_passthrough_from_plain_list(
    sam_vit, mock_sam_vit_patches, fake_pil_image
):
    """invoke handles bounding_boxes that are plain lists (no .tolist) — line 143 else."""
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [np.ones((2, 2), dtype=bool)],
        "scores": torch.tensor([0.5]),
        "bounding_boxes": [[1.0, 2.0, 3.0, 4.0]],  # list, element has no .tolist()
    }
    result = sam_vit.invoke(fake_pil_image)
    assert result["boxes"] == [[1.0, 2.0, 3.0, 4.0]]


def test_invoke_derived_bbox_is_asymmetric_xyxy(
    sam_vit, mock_sam_vit_patches, fake_pil_image
):
    """Derived bbox uses xs for x-extent and ys for y-extent (not swapped).

    The mask spans different ranges in x and y, and min != max on both axes,
    so this kills axis-swap and min/max-flip mutants in the np.where path.
    """
    mask = np.zeros((5, 6), dtype=bool)  # H=5 (rows/y), W=6 (cols/x)
    mask[1:3, 0:4] = True  # rows 1..2 (y), cols 0..3 (x)
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [mask],
        "scores": torch.tensor([0.5]),
        # no "bounding_boxes" -> force derivation
    }
    result = sam_vit.invoke(fake_pil_image)
    # x0=cols.min=0, y0=rows.min=1, x1=cols.max=3, y1=rows.max=2
    assert result["boxes"] == [[0.0, 1.0, 3.0, 2.0]]


def test_invoke_derives_bbox_when_fewer_boxes_than_masks(
    sam_vit, mock_sam_vit_patches, fake_pil_image
):
    """When boxes run out, remaining masks fall back to derivation (i < len bound)."""
    mask0 = np.ones((4, 4), dtype=bool)
    mask1 = np.zeros((4, 4), dtype=bool)
    mask1[0:2, 1:3] = True  # rows 0..1 (y), cols 1..2 (x)
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [mask0, mask1],
        "scores": torch.tensor([0.9, 0.8]),
        "bounding_boxes": torch.tensor([[0.0, 0.0, 3.0, 3.0]]),  # only ONE box
    }
    result = sam_vit.invoke(fake_pil_image)
    assert result["boxes"][0] == [0.0, 0.0, 3.0, 3.0]  # i=0: provided
    assert result["boxes"][1] == [1.0, 0.0, 2.0, 1.0]  # i=1: derived from mask1


def test_invoke_empty_mask_yields_zero_bbox(
    sam_vit, mock_sam_vit_patches, fake_pil_image
):
    """An all-False mask with no provided box yields a [0,0,0,0] bbox — lines 153-154."""
    mock_sam_vit_patches["pipeline"].return_value = {
        "masks": [np.zeros((4, 4), dtype=bool)],
        "scores": torch.tensor([0.1]),
        # no "bounding_boxes" -> derivation; np.where finds nothing
    }
    result = sam_vit.invoke(fake_pil_image)
    assert result["boxes"] == [[0.0, 0.0, 0.0, 0.0]]


# ---------------------------------------------------------------------------
# from_config test
# ---------------------------------------------------------------------------


def test_from_config_constructs_with_yaml_values(mock_sam_vit_patches, tmp_config):
    """from_config reads YAML and passes kwargs to __init__."""
    sam = SamViTModel.from_config(tmp_config)
    mock_sam_vit_patches["SamModel"].from_pretrained.assert_called_once_with(
        "facebook/sam-vit-base", device_map="auto"
    )
    assert sam.points_per_side == 32
    assert sam.pred_iou_thresh == pytest.approx(0.88)
    assert sam.stability_score_thresh == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# LCEL chain composition
# ---------------------------------------------------------------------------


def test_pipe_operator_returns_runnable_sequence(mock_sam_vit_patches):
    """SamViTModel supports | for LCEL chain composition."""
    from langchain_core.runnables import RunnableLambda, RunnableSequence

    sam = SamViTModel(model_id=MODEL_ID)
    chain = sam | RunnableLambda(lambda v: v.get("masks", []))
    assert isinstance(chain, RunnableSequence)


# ---------------------------------------------------------------------------
# _to_bool_array helper
# ---------------------------------------------------------------------------


def test_to_bool_array_from_numpy_array():
    arr = np.array([[True, False], [False, True]])
    result = _to_bool_array(arr)
    assert result.dtype == bool
    np.testing.assert_array_equal(result, arr)


def test_to_bool_array_from_float_numpy_array():
    arr = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = _to_bool_array(arr)
    assert result.dtype == bool
    assert result[0, 0] is np.bool_(True)
    assert result[0, 1] is np.bool_(False)


def test_to_bool_array_from_pil_image():
    img = Image.new("L", (4, 4), color=255)
    result = _to_bool_array(img)
    assert result.dtype == bool
    assert result.all()


def test_to_bool_array_from_pil_image_zeros():
    img = Image.new("L", (4, 4), color=0)
    result = _to_bool_array(img)
    assert result.dtype == bool
    assert not result.any()


def test_to_bool_array_pil_value_127_is_false():
    """A pixel of exactly 127 is below threshold (127 > 127 is False).

    Pins the lower side of the ``> 127`` comparison on line 149.
    """
    img = Image.new("L", (2, 2), color=127)
    result = _to_bool_array(img)
    assert not result.any()


def test_to_bool_array_pil_value_128_is_true():
    """A pixel of exactly 128 is above threshold (128 > 127 is True).

    Pins the upper side of the ``> 127`` comparison on line 149.
    """
    img = Image.new("L", (2, 2), color=128)
    result = _to_bool_array(img)
    assert result.all()


def test_to_bool_array_pil_threshold_direction():
    """The threshold keeps bright pixels and drops dark ones, in that order.

    A gradient straddling the boundary kills comparison-flip mutants
    (e.g. ``>`` → ``<``) that a single solid colour cannot.
    """
    arr = np.array([[0, 127, 128, 255]], dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    result = _to_bool_array(img)
    np.testing.assert_array_equal(result, np.array([[False, False, True, True]]))


def test_to_bool_array_from_torch_tensor():
    """torch.Tensor input is converted to a (H, W) bool numpy array."""
    t = torch.tensor([[1, 0], [0, 1]])
    result = _to_bool_array(t)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    np.testing.assert_array_equal(result, np.array([[True, False], [False, True]]))


def test_to_bool_array_from_torch_bool_tensor():
    """An already-bool torch.Tensor round-trips to the same values."""
    t = torch.tensor([[True, False], [True, True]])
    result = _to_bool_array(t)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    np.testing.assert_array_equal(result, np.array([[True, False], [True, True]]))


def test_to_bool_array_fallback_from_nested_list():
    """A plain (non-numpy, non-PIL, non-tensor) sequence hits the np.asarray fallback."""
    result = _to_bool_array([[1, 0], [0, 1]])
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    np.testing.assert_array_equal(result, np.array([[True, False], [False, True]]))
