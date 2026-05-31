from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from langchain_core.runnables import RunnableConfig
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .base import BaseModel


class GroundingDINOModel(BaseModel):
    """LCEL-compatible wrapper for Grounding DINO (``IDEA-Research/grounding-dino-base``).

    Loads the Grounding DINO object-detection model once at construction time.
    ``invoke`` accepts a dict ``{"image": PIL.Image, "text": str}`` and
    returns the post-processed zero-shot object-detection results.

    The text prompt is a dot-separated list of category names, e.g.
    ``"cat . dog . chair"``.

    Example::

        dino = GroundingDINOModel.from_config("config.yaml")
        result = dino.invoke({"image": pil_img, "text": "laptop . keyboard"})
        boxes  = result["boxes"]   # float tensor [N, 4] in (cx, cy, w, h) format
        scores = result["scores"]  # float tensor [N]
        labels = result["text_labels"]  # list[str] of detected category names
    """

    _config_key = "grounding_dino"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> None:
        """Load the Grounding DINO model and processor.

        The model is loaded **once here**; ``invoke`` never reloads it.

        Args:
            model_id:        HuggingFace model identifier.
            device:          Device string forwarded to HF as ``device_map``
                             (``"auto"``, ``"cuda"``, ``"cpu"``).
            box_threshold:   Minimum objectness score to keep a box.
            text_threshold:  Minimum token-level score to assign a text label.
        """
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, device_map=device
        ).eval()
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    # ------------------------------------------------------------------
    # LCEL entry point
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run Grounding DINO on an image + text prompt and return detections.

        Args:
            input: A dict with keys:

                - ``"image"`` (``PIL.Image.Image``) — the frame to detect objects in.
                - ``"text"`` (``str``) — dot-separated category names, e.g.
                  ``"cat . laptop . chair"``.  A trailing period is added
                  automatically if absent.

            config: Optional LangChain run configuration (unused here).

        Returns:
            The first element of ``post_process_grounded_object_detection``,
            a dict with:

            - ``"boxes"``  — float tensor ``[N, 4]``, each row is
              ``[x0, y0, x1, y1]`` in absolute pixel coordinates.
            - ``"scores"`` — float tensor ``[N]`` of confidence scores.
            - ``"text_labels"`` — list ``[N]`` of matched category names (``str``).

        Raises:
            KeyError: If ``"image"`` or ``"text"`` are missing from *input*.
        """
        img = input["image"]
        text: str = input["text"]

        # Grounding DINO expects the prompt to end with a period.
        if not text.rstrip().endswith("."):
            text = text.rstrip() + " ."

        inp = self.processor(images=img, text=text, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            out = self.model(**inp)

        results: List[Dict[str, Any]] = self.processor.post_process_grounded_object_detection(
            out,
            inp["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]],  # (height, width)
        )
        return results[0]
