"""Tests for src/query/detect.py.

Parametrized so each behavior is exercised twice — once with a SAM-shaped
detector (returns ``masks``) and once with a Grounding-DINO-shaped one
(returns ``labels``).
"""

from __future__ import annotations

import pytest
import torch

from src.data_model import SearchState
from src.query import Detect


# --- detector flavor matrix ------------------------------------------------

SAM_FLAVOR = "sam"
DINO_FLAVOR = "dino"


@pytest.fixture
def detector_flavor(request):
    """Indirect parametrization: yields (flavor_name, mock_detector, build_output)."""
    flavor = request.param
    if flavor == SAM_FLAVOR:
        mock = request.getfixturevalue("mock_sam_model")

        def build(scores):
            return {
                "masks": [torch.zeros(2, 2, dtype=torch.bool)],
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": scores,
            }
    elif flavor == DINO_FLAVOR:
        mock = request.getfixturevalue("mock_dino_model")

        def build(scores):
            return {
                "labels": ["object"],
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": scores,
            }
    else:
        raise ValueError(flavor)

    return flavor, mock, build


# --- core behaviors (run on both detectors) --------------------------------


@pytest.mark.parametrize(
    "detector_flavor", [SAM_FLAVOR, DINO_FLAVOR], indirect=True
)
def test_invoke_sets_detected_with_max_score(detector_flavor, retrieved_from):
    flavor, mock, build = detector_flavor
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
        return build(score_seq[i])

    mock.invoke.side_effect = _invoke

    state = SearchState(query="q", collection_id="c", retrieved=retrieved)
    out = Detect(mock).invoke(state)

    assert out.detected is not None and len(out.detected) == 3
    # detection_score is max(scores) per image.
    assert out.detected[0].detection_score == pytest.approx(0.4)
    assert out.detected[1].detection_score == pytest.approx(0.9)
    assert out.detected[2].detection_score == pytest.approx(0.5)
    # Order preserved (no sorting in this step).
    assert [d.id for d in out.detected] == [r.id for r in retrieved]


@pytest.mark.parametrize(
    "detector_flavor", [SAM_FLAVOR, DINO_FLAVOR], indirect=True
)
def test_invoke_handles_empty_score_tensor(detector_flavor, retrieved_from):
    _, mock, _ = detector_flavor
    retrieved = retrieved_from(1)
    mock.invoke.side_effect = lambda _: {
        "boxes": torch.zeros(0, 4),
        "scores": torch.zeros(0),
    }

    state = SearchState(query="q", collection_id="c", retrieved=retrieved)
    out = Detect(mock).invoke(state)

    assert out.detected[0].detection_score == 0.0


@pytest.mark.parametrize(
    "detector_flavor", [SAM_FLAVOR, DINO_FLAVOR], indirect=True
)
def test_invoke_passes_query_as_prompt(detector_flavor, retrieved_from):
    _, mock, _ = detector_flavor
    retrieved = retrieved_from(2)
    state = SearchState(query="laptop", collection_id="c", retrieved=retrieved)
    Detect(mock).invoke(state)

    for call in mock.invoke.call_args_list:
        (payload,) = call.args
        assert payload["text"] == "laptop"
        assert payload["image"] is not None


@pytest.mark.parametrize(
    "detector_flavor", [SAM_FLAVOR, DINO_FLAVOR], indirect=True
)
def test_invoke_rejects_missing_retrieved(detector_flavor):
    _, mock, _ = detector_flavor
    state = SearchState(query="q", collection_id="c")
    with pytest.raises(ValueError, match="retrieved"):
        Detect(mock).invoke(state)


# --- detector-specific output keys -----------------------------------------


def test_sam_path_populates_masks_not_labels(mock_sam_model, retrieved_from):
    retrieved = retrieved_from(1)
    state = SearchState(query="q", collection_id="c", retrieved=retrieved)
    out = Detect(mock_sam_model).invoke(state)
    d = out.detected[0]
    assert d.masks is not None
    assert d.labels is None


def test_dino_path_populates_labels_not_masks(mock_dino_model, retrieved_from):
    retrieved = retrieved_from(1)
    state = SearchState(query="q", collection_id="c", retrieved=retrieved)
    out = Detect(mock_dino_model).invoke(state)
    d = out.detected[0]
    assert d.labels == ["object"]
    assert d.masks is None
