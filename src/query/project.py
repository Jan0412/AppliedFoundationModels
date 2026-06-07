"""Step 5 — back-project the final 2D results and fuse them into one 3D object."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional, Tuple

import numpy as np
import torch
from langchain_core.runnables import Runnable, RunnableConfig
from PIL import Image

from src.data_model import ProjectedObject, SearchState
from src.utils.db import load_collection_meta
from src.utils.geometry import (
    backproject,
    box_to_mask,
    largest_cluster,
    robust_bounds,
    voxel_downsample,
)

logger = logging.getLogger(__name__)


class ProjectTo3D(Runnable):
    """Back-project the final results and fuse them into one tight 3D object.

    Each hit contributes pixels — the best **mask** if the detector produced
    masks (SAM), otherwise the best 2D **box** rasterised to a rectangle (e.g.
    Grounding DINO) — which are back-projected to world space using the per-row
    ``cam2world`` pose and the per-collection intrinsics + ``depth_scale`` (read
    from the LanceDB metadata table written at index time).

    Points from **all** hits are then merged and run through DBSCAN. The object
    is consistent across frames and forms the dominant dense cluster, while the
    per-frame background (different from each viewpoint) scatters into noise and
    is dropped. This is what lets a box-only detector still yield a *tight* box:
    multi-frame consensus separates object from background. The largest cluster
    becomes a single fused :class:`ProjectedObject` (cleaned cloud + axis-aligned
    world box).

    The collection must have been indexed with depth + poses + calibration.

    Pre:  ``state.results`` or ``state.detected`` is set; the collection has a
          ``_collection_meta`` row.
    Post: ``state.projected`` holds a **single** fused :class:`ProjectedObject`
          (or is empty when nothing could be back-projected).
    """

    def __init__(
        self,
        db,
        voxel: float = 0.02,
        cluster_eps: float = 0.05,
        cluster_min_samples: int = 10,
        bbox_percentile: Tuple[float, float] = (2.0, 98.0),
    ) -> None:
        self.db = db
        self.voxel = voxel
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.bbox_percentile = bbox_percentile
        self._meta_cache: dict[str, dict] = {}

    def _meta(self, collection_id: str) -> dict:
        if collection_id not in self._meta_cache:
            meta = load_collection_meta(self.db, collection_id)
            if meta is None:
                raise ValueError(
                    f"ProjectTo3D: no calibration found for collection "
                    f"'{collection_id}'. Re-index it with intrinsics + "
                    "depth_scale (see Indexer.insert) before projecting."
                )
            self._meta_cache[collection_id] = meta
        return self._meta_cache[collection_id]

    @staticmethod
    def _select_mask(hit) -> Optional[np.ndarray]:
        """Return the boolean source mask for *hit*, or ``None`` if it has none.

        Prefers the highest-scoring segmentation mask; falls back to rasterising
        the highest-scoring 2D box (so box-only detectors are supported).
        """
        if hit.masks:
            best = int(torch.as_tensor(hit.scores).argmax()) if len(hit.masks) > 1 else 0
            mask = hit.masks[best]
            return mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)

        boxes = getattr(hit, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
        best = int(torch.as_tensor(hit.scores).argmax()) if len(boxes_np) > 1 else 0
        # Box coordinates are in detection (RGB) space; backproject resizes the
        # mask to depth resolution when they differ.
        w, h = Image.open(hit.path).size
        return box_to_mask(boxes_np[best], h, w)

    def invoke(
        self,
        state: SearchState,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> SearchState:
        hits = state.results if state.results is not None else state.detected
        if hits is None:
            raise ValueError(
                "ProjectTo3D: state.results/state.detected is None — run the "
                "detect (and rerank) steps first."
            )

        meta = self._meta(state.collection_id)
        intr = (meta["fx"], meta["fy"], meta["cx"], meta["cy"])
        depth_scale = meta["depth_scale"]

        # 1. Back-project every hit and accumulate one big point cloud.
        pts_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        have_colors = True
        for hit in hits:
            if not hit.depth_path or hit.cam2world is None:
                warnings.warn(
                    f"ProjectTo3D: skipping '{hit.id}' — missing depth_path or "
                    "cam2world pose.",
                    stacklevel=2,
                )
                continue
            mask_np = self._select_mask(hit)
            if mask_np is None:
                warnings.warn(
                    f"ProjectTo3D: skipping '{hit.id}' — no masks or boxes to "
                    "back-project.",
                    stacklevel=2,
                )
                continue
            points, colors = backproject(
                mask_np, hit.depth_path, hit.cam2world, intr, depth_scale,
                rgb_path=hit.path or None,
            )
            if len(points) == 0:
                continue
            pts_parts.append(points)
            if colors is None:
                have_colors = False
            else:
                col_parts.append(colors)

        if not pts_parts:
            warnings.warn(
                "ProjectTo3D: no 3D points to fuse (no detections with valid "
                "depth).",
                stacklevel=2,
            )
            return state.model_copy(update={"projected": []})

        all_points = np.concatenate(pts_parts)
        all_colors = np.concatenate(col_parts) if (have_colors and col_parts) else None

        # 2. Even out density, then keep the dominant (object) cluster.
        ds_points, ds_colors = voxel_downsample(all_points, all_colors, self.voxel)
        mask = largest_cluster(ds_points, self.cluster_eps, self.cluster_min_samples)
        obj_points = ds_points[mask]
        obj_colors = ds_colors[mask] if ds_colors is not None else None

        # 3. One fused object: cleaned cloud + tight axis-aligned world box.
        bbox = robust_bounds(obj_points, *self.bbox_percentile)
        fused = ProjectedObject(
            id="fused", path="", points=obj_points, colors=obj_colors, bbox=bbox
        )
        return state.model_copy(update={"projected": [fused]})
