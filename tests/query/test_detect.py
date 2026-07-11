"""Tests for src/query/detect.py."""

from __future__ import annotations

import pytest
import torch

from src.data_model import SearchState
from src.query import Detect


def _sam_output(scores):
    return {
        "masks": [torch.zeros(2, 2, dtype=torch.bool)],
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "scores": scores,
    }


def test_invoke_sets_detected_with_max_score(mock_sam_model, retrieved_from):
    retrieved = retrieved_from(3)

    score_seq = [
        torch.tensor([0.1, 0.4]),
        torch.tensor([0.9, 0.2]),
        torch.tensor([0.5]),
    ]
    call_idx = {"i": 0}

    def _invoke(_):
        i = call_idx["i"]
        call_idx["i"] += 1
        return _sam_output(score_seq[i])

    mock_sam_model.invoke.side_effect = _invoke

    state = SearchState(query="q", collection_id="c", retrieved=retrieved)
    out = Detect(mock_sam_model).invoke(state)

    assert out.detected is not None and len(out.detected) == 3
    # detection_score is max(scores) per image.
    assert out.detected[0].detection_score == pytest.approx(0.4)
    assert out.detected[1].detection_score == pytest.approx(0.9)
    assert out.detected[2].detection_score == pytest.approx(0.5)
    # Order preserved (no sorting in this step).
    assert [d.id for d in out.detected] == [r.id for r in retrieved]


def test_invoke_handles_empty_score_tensor(mock_sam_model, retrieved_from):
    retrieved = retrieved_from(1)
    mock_sam_model.invoke.side_effect = lambda _: {
        "boxes": torch.zeros(0, 4),
        "scores": torch.zeros(0),
    }

    state = SearchState(query="q", collection_id="c", retrieved=retrieved)
    out = Detect(mock_sam_model).invoke(state)

    assert out.detected[0].detection_score == 0.0


def test_invoke_passes_query_as_prompt(mock_sam_model, retrieved_from):
    retrieved = retrieved_from(2)
    state = SearchState(query="laptop", collection_id="c", retrieved=retrieved)
    Detect(mock_sam_model).invoke(state)

    for call in mock_sam_model.invoke.call_args_list:
        (payload,) = call.args
        assert payload["text"] == "laptop"
        assert payload["image"] is not None


def test_invoke_rejects_missing_retrieved(mock_sam_model):
    state = SearchState(query="q", collection_id="c")
    with pytest.raises(ValueError, match="retrieved"):
        Detect(mock_sam_model).invoke(state)


def test_lazy_loads_unloaded_images_from_path(mock_sam_model, tiny_image_files):
    from PIL import Image

    from src.data_model import RetrievedImage

    (path,) = tiny_image_files(1)
    retrieved = [
        RetrievedImage(id="id-0", path=path, similarity_score=0.9, image=None)
    ]
    state = SearchState(query="q", collection_id="c", retrieved=retrieved)

    Detect(mock_sam_model).invoke(state)

    (payload,) = mock_sam_model.invoke.call_args.args
    assert isinstance(payload["image"], Image.Image)
    assert payload["image"].mode == "RGB"
    assert retrieved[0].image is payload["image"]  # cached for later steps


def test_unloaded_image_without_path_raises(mock_sam_model):
    from src.data_model import RetrievedImage

    retrieved = [
        RetrievedImage(id="id-0", path="", similarity_score=0.9, image=None)
    ]
    state = SearchState(query="q", collection_id="c", retrieved=retrieved)

    with pytest.raises(ValueError, match="no cached"):
        Detect(mock_sam_model).invoke(state)


def test_masks_populated_when_detector_returns_them(mock_sam_model, retrieved_from):
    retrieved = retrieved_from(1)
    state = SearchState(query="q", collection_id="c", retrieved=retrieved)
    out = Detect(mock_sam_model).invoke(state)
    assert out.detected[0].masks is not None


def test_masks_none_when_detector_omits_them(mock_sam_model, retrieved_from):
    """Detect is duck-typed: a detector returning no masks leaves the slot None."""
    retrieved = retrieved_from(1)
    mock_sam_model.invoke.side_effect = lambda _: {
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "scores": torch.tensor([0.5]),
    }
    state = SearchState(query="q", collection_id="c", retrieved=retrieved)
    out = Detect(mock_sam_model).invoke(state)
    assert out.detected[0].masks is None
