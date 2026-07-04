"""Step 3 — run a detector (SAM, Grounding DINO, …) on each retrieved image."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import Runnable, RunnableConfig
from tqdm import tqdm

from src.data_model import DetectedImage, SearchState


class Detect(Runnable):
    """Run a generic detector on every ``RetrievedImage`` with the query.

    The detector is **duck-typed**: anything whose
    ``invoke({"image": pil, "text": str})`` returns a dict containing at
    least ``scores`` and ``boxes`` slots in (both :class:`SAMModel` and
    :class:`GroundingDINOModel` satisfy this).

    Optional output keys are copied through to the resulting
    :class:`DetectedImage`:

    - ``masks``  — SAM only; stays ``None`` when absent.
    - ``labels`` — Grounding DINO only; stays ``None`` when absent.

    ``detection_score`` is ``float(scores.max())`` when the score tensor
    is non-empty, else ``0.0`` (so images the detector couldn't ground
    fall to the bottom but are not silently dropped — that's the rerank
    step's job).

    Pre:  ``state.retrieved`` is set. Images may still be unloaded — they
          are loaded (and cached) on demand via ``RetrievedImage.load_image``.
    Post: ``state.detected`` is a list of :class:`DetectedImage`, same
          order as ``state.retrieved`` (no sorting, no trimming).
    """

    def __init__(self, detector) -> None:
        self.detector = detector

    def invoke(
        self,
        state: SearchState,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> SearchState:
        if state.retrieved is None:
            raise ValueError(
                "Detect: state.retrieved is None — run RetrieveSimilar first."
            )

        detected: list[DetectedImage] = []
        for ri in tqdm(state.retrieved, desc="Segmenting frames", unit="frame"):
            out = self.detector.invoke({"image": ri.load_image(), "text": state.query})
            scores = out["scores"].cpu()
            score = float(scores.max()) if scores.numel() > 0 else 0.0
            detected.append(
                DetectedImage(
                    id=ri.id,
                    path=ri.path,
                    similarity_score=ri.similarity_score,
                    detection_score=score,
                    boxes=out["boxes"].cpu(),
                    scores=scores,
                    masks=[m.cpu() for m in out["masks"]] if "masks" in out else None,
                    labels=list(out["labels"]) if "labels" in out else None,
                    depth_path=ri.depth_path,
                    cam2world=ri.cam2world,
                )
            )

        return state.model_copy(update={"detected": detected})
