from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
import yaml
from PIL import Image

from src.models.sam_vit import SamViTModel

logger = logging.getLogger(__name__)

_VALID_BACKGROUNDS = ("gray", "black", "white", "noise")


class ObjectExtractor:
    """Remove the background from an image and return centred object crops.

    Uses :class:`~src.models.sam_vit.SamViTModel` to segment every object in
    the image automatically, ranks the detected masks by ``area × centrality``,
    keeps the top-*top_n* ones, fills the background with the chosen fill mode,
    crops each to its bounding box, centres it on a square canvas, and resizes
    to *output_size* × *output_size*.

    The resulting images are ready to pass directly to SigLIP for embedding —
    each object fills its frame rather than appearing as a small element in a
    cluttered scene.

    Supported background fill modes:

    - ``"gray"``  — neutral mid-gray (128, 128, 128); recommended for
      CLIP/SigLIP since it injects the least spurious signal.
    - ``"black"`` — pure black (0, 0, 0).
    - ``"white"`` — pure white (255, 255, 255).
    - ``"noise"`` — Gaussian noise (mean/std configurable).  Expected to
      perform worst for SigLIP due to high-frequency contamination; provided
      for comparison.

    Example::

        pre = ObjectExtractor.from_config("config.yaml")
        crops = pre.extract(raw_pil_image)          # List[PIL.Image] of length top_n
        all_crops = pre.extract(list_of_images)     # List[List[PIL.Image]]
        idx.insert(
            [c for crops in all_crops for c in crops],   # flatten
            collection_id="fr1_desk", ids=[...],
        )
    """

    def __init__(
        self,
        sam: SamViTModel,
        background: str = "gray",
        selection: str = "area_centrality",
        margin: float = 0.1,
        output_size: int = 224,
        top_n: int = 1,
        noise_mean: float = 128.0,
        noise_std: float = 40.0,
    ) -> None:
        """
        Args:
            sam:         Pre-loaded :class:`SamViTModel`; ``invoke`` is called
                         once per image.
            background:  Background fill mode.  One of ``"gray"``, ``"black"``,
                         ``"white"``, ``"noise"``.
            selection:   Mask selection strategy.  Currently only
                         ``"area_centrality"`` (area × centrality score) is
                         supported.
            margin:      Bounding-box expansion as a fraction of bbox edge
                         length (e.g. ``0.1`` → 10 % padding on each side).
            output_size: Edge length of the returned square PIL image, in
                         pixels.  Should match the embedding model's expected
                         input size (224 for most SigLIP variants).
            top_n:       Number of object crops to return per image, ranked
                         best-first by area × centrality.  ``1`` returns only
                         the dominant object.  When fewer than *top_n* valid
                         masks exist the list is shorter rather than padded.
            noise_mean:  Mean pixel value for Gaussian noise background
                         (0–255 scale).  Only used when
                         ``background == "noise"``.
            noise_std:   Standard deviation for Gaussian noise background.
        """
        if background not in _VALID_BACKGROUNDS:
            raise ValueError(
                f"background={background!r} is not valid; "
                f"choose one of {_VALID_BACKGROUNDS}."
            )
        self.sam = sam
        self.background = background
        self.selection = selection
        self.margin = margin
        self.output_size = output_size
        self.top_n = top_n
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    # ------------------------------------------------------------------
    # Construction from config.yaml
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, path: str | Path = "config.yaml") -> "ObjectExtractor":
        """Build an :class:`ObjectExtractor` from *path* (a YAML config file).

        Reads ``preprocess:`` for the extractor kwargs and delegates to
        :meth:`SamViTModel.from_config` for the SAM model.

        Example::

            pre = ObjectExtractor.from_config("config.yaml")
        """
        cfg = yaml.safe_load(Path(path).read_text())
        sam = SamViTModel.from_config(path)
        pre_cfg: dict = cfg.get("preprocess", {})
        return cls(
            sam=sam,
            background=pre_cfg.get("background", "gray"),
            selection=pre_cfg.get("selection", "area_centrality"),
            margin=float(pre_cfg.get("margin", 0.1)),
            output_size=int(pre_cfg.get("output_size", 224)),
            top_n=int(pre_cfg.get("top_n", 1)),
            noise_mean=float(pre_cfg.get("noise_mean", 128.0)),
            noise_std=float(pre_cfg.get("noise_std", 40.0)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        input: Union[Image.Image, List[Image.Image]],
    ) -> Union[List[Image.Image], List[List[Image.Image]]]:
        """Extract background-removed, centred object crops from *input*.

        Args:
            input: A single ``PIL.Image.Image`` or a list of them.

        Returns:
            - **Single image** → ``List[PIL.Image.Image]`` of length
              1..*top_n* (one crop per selected object, best first).
            - **List of images** → ``List[List[PIL.Image.Image]]``, one inner
              list per input image.

        Raises:
            TypeError: If *input* is not a PIL image or list of PIL images.
        """
        if isinstance(input, list):
            return [self._process_one(img) for img in input]
        if isinstance(input, Image.Image):
            return self._process_one(input)
        raise TypeError(
            f"ObjectExtractor.extract: expected PIL.Image.Image or list thereof, "
            f"got {type(input).__name__!r}."
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _process_one(self, img: Image.Image) -> List[Image.Image]:
        """Return up to *top_n* background-removed crops for a single image."""
        img_rgb = img.convert("RGB")
        result = self.sam.invoke(img_rgb)

        masks: list = result.get("masks", [])
        if not masks:
            warnings.warn(
                "SamViTModel returned no masks; returning original image unchanged.",
                stacklevel=2,
            )
            return [img_rgb.resize((self.output_size, self.output_size), Image.LANCZOS)]

        top_indices = self._select_top_n_masks(masks, img_rgb)

        crops: List[Image.Image] = []
        for best_idx in top_indices:
            mask = np.asarray(masks[best_idx], dtype=bool)
            arr = np.array(img_rgb)
            out = self._apply_background(arr, mask)

            ys, xs = np.where(mask)
            if len(ys) == 0:
                warnings.warn(
                    "Selected mask is empty; returning original image unchanged.",
                    stacklevel=2,
                )
                crops.append(
                    img_rgb.resize((self.output_size, self.output_size), Image.LANCZOS)
                )
                continue

            h, w = mask.shape
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

            # Expand bbox by margin fraction
            dx = max(1, int((x1 - x0) * self.margin))
            dy = max(1, int((y1 - y0) * self.margin))
            x0 = max(0, x0 - dx)
            y0 = max(0, y0 - dy)
            x1 = min(w, x1 + dx)
            y1 = min(h, y1 + dy)

            crop = out[y0:y1, x0:x1]

            # Centre crop on a square canvas the size of its longest edge
            ch, cw = crop.shape[:2]
            side = max(ch, cw)
            canvas = self._make_canvas(side, side)
            pad_x = (side - cw) // 2
            pad_y = (side - ch) // 2
            canvas[pad_y:pad_y + ch, pad_x:pad_x + cw] = crop

            crops.append(
                Image.fromarray(canvas).resize(
                    (self.output_size, self.output_size), Image.LANCZOS
                )
            )

        return crops

    def _select_top_n_masks(self, masks: list, img: Image.Image) -> List[int]:
        """Return the indices of the top-``top_n`` masks, ranked by area × centrality.

        Masks are scored as ``(pixel_count / total_pixels) × (1 - normalised
        distance of centroid from image centre)``.  Empty (all-False) masks are
        skipped.  If all masks are empty, ``[0]`` is returned as a safe fallback.
        The returned list is shorter than *top_n* when fewer valid masks exist.
        """
        w, h = img.size
        cx, cy = w / 2.0, h / 2.0
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        total_px = float(h * w)

        scored: list = []
        for i, raw in enumerate(masks):
            m = np.asarray(raw, dtype=bool)
            count = m.sum()
            if count == 0:
                continue
            area = count / total_px
            ys, xs = np.where(m)
            mcx, mcy = xs.mean(), ys.mean()
            dist = np.sqrt((mcx - cx) ** 2 + (mcy - cy) ** 2)
            centrality = 1.0 - dist / max_dist if max_dist > 0 else 1.0
            scored.append((area * centrality, i))

        if not scored:
            return [0]

        scored.sort(reverse=True)
        return [idx for _, idx in scored[: self.top_n]]

    def _apply_background(self, arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return *arr* with background (where *mask* is False) filled."""
        out = arr.copy()
        bg = ~mask
        if self.background == "gray":
            out[bg] = (128, 128, 128)
        elif self.background == "black":
            out[bg] = (0, 0, 0)
        elif self.background == "white":
            out[bg] = (255, 255, 255)
        else:  # "noise"
            rng = np.random.default_rng()
            n_bg = int(bg.sum())
            noise = rng.normal(self.noise_mean, self.noise_std, (n_bg, 3))
            out[bg] = np.clip(noise, 0, 255).astype(np.uint8)
        return out

    def _make_canvas(self, h: int, w: int) -> np.ndarray:
        """Return an (h, w, 3) uint8 array filled with the background colour."""
        if self.background == "gray":
            return np.full((h, w, 3), 128, dtype=np.uint8)
        if self.background == "black":
            return np.zeros((h, w, 3), dtype=np.uint8)
        if self.background == "white":
            return np.full((h, w, 3), 255, dtype=np.uint8)
        # noise
        rng = np.random.default_rng()
        noise = rng.normal(self.noise_mean, self.noise_std, (h, w, 3))
        return np.clip(noise, 0, 255).astype(np.uint8)
