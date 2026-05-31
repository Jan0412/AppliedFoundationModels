from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from langchain_core.runnables import RunnableConfig
from transformers import Sam3Model, Sam3Processor

from .base import BaseModel


class SAMModel(BaseModel):
    """LCEL-compatible wrapper for SAM3 (``facebook/sam3``).

    Loads the SAM3 segmentation model once at construction time.
    ``invoke`` accepts a dict ``{"image": PIL.Image, "text": str}`` and
    returns the post-processed segmentation result for that image/prompt
    pair.

    Example::

        sam = SAMModel.from_config("config.yaml")
        result = sam.invoke({"image": pil_img, "text": "laptop"})
        masks  = result["masks"]   # list of boolean tensors
        scores = result["scores"]  # confidence scores
        boxes  = result["boxes"]   # bounding boxes
    """

    _config_key = "sam"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> None:
        """Load the SAM3 model and processor.

        The model is loaded **once here**; ``invoke`` never reloads it.

        Args:
            model_id:       HuggingFace model identifier.
            device:         Device string forwarded to HF as ``device_map``
                            (``"auto"``, ``"cuda"``, ``"cpu"``).
            threshold:      Minimum confidence score to keep a segment.
            mask_threshold: Mask binarisation threshold.
        """
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, device_map=device).eval()
        self.device = device
        self.threshold = threshold
        self.mask_threshold = mask_threshold

    # ------------------------------------------------------------------
    # LCEL entry point
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run SAM3 on an image + text prompt and return segmentation results.

        Args:
            input: A dict with keys:

                - ``"image"`` (``PIL.Image.Image``) — the frame to segment.
                - ``"text"`` (``str``) — the text prompt describing the
                  object(s) to highlight.

            config: Optional LangChain run configuration (unused here).

        Returns:
            The first element of ``post_process_instance_segmentation``,
            a dict with:

            - ``"masks"``  — list of boolean tensors, one per segment.
            - ``"scores"`` — float tensor of confidence scores.
            - ``"boxes"``  — float tensor of bounding boxes ``[x0, y0, x1, y1]``.

        Raises:
            KeyError: If ``"image"`` or ``"text"`` are missing from *input*.
        """
        img = input["image"]
        text = input["text"]

        inp = self.processor(images=img, text=text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inp)

        return self.processor.post_process_instance_segmentation(
            out,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inp.get("original_sizes").tolist(),
        )[0]
