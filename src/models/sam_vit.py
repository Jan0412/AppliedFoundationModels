from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from langchain_core.runnables import RunnableConfig
from transformers import SamModel, SamProcessor, pipeline

from .base import BaseModel


class SamViTModel(BaseModel):
    """LCEL-compatible wrapper for the original Segment Anything Model (ViT backbone).

    Loads the SAM model once at construction time and builds a HuggingFace
    ``mask-generation`` pipeline for automatic, prompt-free segmentation.
    ``invoke`` accepts a single ``PIL.Image.Image`` and returns every detected
    object mask — no text prompt required.

    This is distinct from :class:`~src.models.sam.SAMModel`, which wraps the
    newer text-promptable SAM3.  Use ``SamViTModel`` when you want to segment
    *all* objects in an image (e.g. for preprocessing before embedding).

    Example::

        sam = SamViTModel.from_config("config.yaml")
        result = sam.invoke(pil_img)
        masks  = result["masks"]   # list[np.ndarray (H, W)] bool
        scores = result["scores"]  # list[float] predicted-IoU
        boxes  = result["boxes"]   # list[[x0, y0, x1, y1]] pixel coords
    """

    _config_key = "sam_vit"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ) -> None:
        """Load the SAM ViT model and build the mask-generation pipeline.

        The model is loaded **once here**; ``invoke`` never reloads it.

        Args:
            model_id:                HuggingFace model identifier,
                                     e.g. ``"facebook/sam-vit-base"``.
            device:                  Device string forwarded to HF as
                                     ``device_map`` (``"auto"``, ``"cuda"``,
                                     ``"cpu"``).
            points_per_side:         Number of points sampled per side of the
                                     image grid for automatic mask generation.
            points_per_batch:        Number of grid points processed per
                                     forward pass; lower values reduce peak
                                     VRAM usage.
            pred_iou_thresh:         Minimum predicted-IoU to keep a mask.
            stability_score_thresh:  Minimum stability score to keep a mask.
        """
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id, device_map=device).eval()
        self.device = device
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh

        # Build mask-generation pipeline reusing the already-loaded model so we
        # don't pay the download/load cost twice.
        self.mask_generator = pipeline(
            task="mask-generation",
            model=self.model,
            image_processor=self.processor.image_processor,
        )

    # ------------------------------------------------------------------
    # LCEL entry point
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Image.Image,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run automatic mask generation on *input* and return all object masks.

        Args:
            input:  A ``PIL.Image.Image`` to segment.
            config: Optional LangChain run configuration (unused here).

        Returns:
            A dict with:

            - ``"masks"``  — ``list[np.ndarray]``, each boolean array of
              shape ``(H, W)``; ``True`` where the object is.
            - ``"scores"`` — ``list[float]`` predicted-IoU for each mask.
            - ``"boxes"``  — ``list[[x0, y0, x1, y1]]`` in absolute pixel
              coordinates (float).

        Raises:
            TypeError: If *input* is not a ``PIL.Image.Image``.
        """
        if not isinstance(input, Image.Image):
            raise TypeError(
                f"SamViTModel.invoke: expected PIL.Image.Image, "
                f"got {type(input).__name__!r}."
            )

        # transformers 5.x _sanitize_parameters maps `points_per_crop` to the
        # grid-density parameter (equivalent to `points_per_side` in orig SAM).
        outputs = self.mask_generator(
            input,
            points_per_crop=self.points_per_side,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            output_bboxes_mask=True,
        )

        # Pipeline returns a dict in transformers 5.x:
        #   {"masks": List[np.ndarray (H,W) bool], "scores": Tensor,
        #    "bounding_boxes": Tensor (N,4) XYXY}
        raw_masks  = outputs.get("masks", [])
        raw_scores = outputs.get("scores", [])
        raw_boxes  = outputs.get("bounding_boxes")

        masks:  List[np.ndarray]   = []
        scores: List[float]        = []
        boxes:  List[List[float]]  = []

        for i, seg in enumerate(raw_masks):
            bool_mask = _to_bool_array(seg)
            masks.append(bool_mask)
            scores.append(float(raw_scores[i]))

            if raw_boxes is not None and i < len(raw_boxes):
                box = raw_boxes[i]
                coords = box.tolist() if hasattr(box, "tolist") else list(box)
                boxes.append([float(v) for v in coords])  # already XYXY
            else:
                # Derive XYXY bbox from mask pixels when boxes are absent
                ys, xs = np.where(bool_mask)
                if len(ys) > 0:
                    boxes.append([
                        float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max()),
                    ])
                else:
                    boxes.append([0.0, 0.0, 0.0, 0.0])

        return {"masks": masks, "scores": scores, "boxes": boxes}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _to_bool_array(seg: Any) -> np.ndarray:
    """Normalise a mask from the pipeline into a (H, W) bool numpy array."""
    if isinstance(seg, np.ndarray):
        return seg.astype(bool)
    if isinstance(seg, Image.Image):
        return np.array(seg.convert("L")) > 127
    if isinstance(seg, torch.Tensor):
        return seg.bool().cpu().numpy()
    return np.asarray(seg, dtype=bool)
