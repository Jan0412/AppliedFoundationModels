"""Step 5 — back-project the final 2D results and fuse them into one 3D object."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional, Tuple

import numpy as np
import torch
from langchain_core.runnables import Runnable, RunnableConfig

from src.data_model import ProjectedObject, SearchState
from src.utils.db import load_collection_meta
from src.utils.geometry import (
    backproject,
    cluster_masks,
    largest_cluster,
    robust_bounds,
    voxel_downsample,
)

logger = logging.getLogger(__name__)

#: Valid projection modes (see :class:`ProjectTo3D`).
_MODES = ("simple", "cluster_single", "cluster_instances")


class ProjectTo3D(Runnable):
    """Back-project the final results and fuse them into 3D object(s).

    Each hit contributes **mask** pixels (produced by SAM3): the single
    highest-scoring mask for ``simple``/``cluster_single``, or the union of
    **all** the frame's masks for ``cluster_instances`` (so every detected
    segment is back-projected, not just the top one). Those pixels are
    back-projected to world space using the per-row ``cam2world`` pose and the
    per-collection intrinsics + ``depth_scale`` (read from the LanceDB metadata
    table written at index time).

    Points from **all** hits are merged, then handled per ``mode`` (constructor
    default, overridable per query via ``state.mode``):

    - ``"simple"`` — no clustering. Every back-projected point is kept as one
      fused :class:`ProjectedObject` (``id="fused"``). No noise rejection or
      instance splitting, so the box is looser but nothing is dropped.
    - ``"cluster_single"`` — DBSCAN, then the largest cluster becomes one fused
      :class:`ProjectedObject` (``id="fused"``). DBSCAN groups the object (dense,
      consistent across frames) and drops the per-frame background (scattered
      into noise): multi-frame consensus separates object from background. The
      right mode when the query targets one unique object (e.g. ScanRefer).
    - ``"cluster_instances"`` — back-projects **every** SAM mask per frame
      (not just the best one), then DBSCAN, then every cluster with at least
      ``min_instance_size`` points becomes its own :class:`ProjectedObject`
      (``id="obj-{i}"``), sorted by point count descending — the most
      multi-frame consensus (most probable object) first, so ``projected[0]``
      matches ``cluster_single``. Use this when several distinct objects
      (e.g. multiple chairs/books) should each be recovered: SAM's per-frame
      segments feed the cloud and 3D clustering splits them into instances.

      Because every segment is back-projected, a false positive that SAM
      grounds in a *single* frame lands as a dense, coherent blob — density
      alone no longer distinguishes it from a real object. ``min_views``
      restores the multi-frame consensus that top-1 selection used to give for
      free: a cluster is kept only when its points come from at least that many
      distinct frames, so single-frame flukes are dropped while genuinely
      multi-view objects survive.

    The collection must have been indexed with depth + poses + calibration.

    Pre:  ``state.results`` or ``state.detected`` is set; the collection has a
          ``_collection_meta`` row.
    Post: ``state.projected`` holds the fused :class:`ProjectedObject` (s)
          (or is empty when nothing could be back-projected).
    """

    def __init__(
        self,
        db,
        voxel: float = 0.02,
        mode: str = "cluster_single",
        cluster_eps: float = 0.05,
        cluster_min_samples: int = 10,
        bbox_percentile: Tuple[float, float] = (2.0, 98.0),
        max_instances: Optional[int] = None,
        min_instance_size: int = 50,
        min_views: int = 1,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(
                f"ProjectTo3D: unknown mode {mode!r}. "
                f"Expected one of {sorted(_MODES)}."
            )
        self.db = db
        self.voxel = voxel
        self.mode = mode
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.bbox_percentile = bbox_percentile
        self.max_instances = max_instances
        self.min_instance_size = min_instance_size
        self.min_views = min_views
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
    def _to_np(mask) -> np.ndarray:
        return mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)

    @classmethod
    def _select_mask(cls, hit, all_masks: bool = False) -> Optional[np.ndarray]:
        """Return the boolean source mask for *hit*, or ``None`` if it has none.

        With ``all_masks`` False (default) uses the single highest-scoring
        segmentation mask — the right choice when the query targets one object
        (``simple``/``cluster_single``). With ``all_masks`` True the union of
        **every** SAM mask for the frame is returned, so all detected segments
        contribute pixels and instance splitting is left to 3D clustering
        (``cluster_instances``). SAM3 always produces masks.
        """
        if not hit.masks:
            return None
        if all_masks:
            union = cls._to_np(hit.masks[0]).astype(bool)
            for m in hit.masks[1:]:
                union |= cls._to_np(m).astype(bool)
            return union
        best = int(torch.as_tensor(hit.scores).argmax()) if len(hit.masks) > 1 else 0
        return cls._to_np(hit.masks[best])

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

        # cluster_instances back-projects every SAM segment per frame (3D
        # clustering splits them); the single-object modes use the best mask.
        mode = state.mode or self.mode
        all_masks = mode == "cluster_instances"

        # 1. Back-project every hit and accumulate one big point cloud. Each
        # point remembers the frame it came from, so clusters can be scored on
        # how many distinct frames back them (see min_views below).
        pts_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        frame_parts: list[np.ndarray] = []
        have_colors = True
        for frame_idx, hit in enumerate(hits):
            if not hit.depth_path or hit.cam2world is None:
                warnings.warn(
                    f"ProjectTo3D: skipping '{hit.id}' — missing depth_path or "
                    "cam2world pose.",
                    stacklevel=2,
                )
                continue
            mask_np = self._select_mask(hit, all_masks=all_masks)
            if mask_np is None:
                warnings.warn(
                    f"ProjectTo3D: skipping '{hit.id}' — no mask to "
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
            frame_parts.append(np.full(len(points), frame_idx, dtype=np.int64))
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
        all_frames = np.concatenate(frame_parts)

        # 2. Even out density before clustering. `inverse` maps every original
        # point to the kept point representing its voxel, so the per-point frame
        # ids survive the downsample (a kept point alone only carries the frame
        # of whichever source point happened to represent its voxel).
        ds_points, ds_colors, inverse = voxel_downsample(
            all_points, all_colors, self.voxel, return_inverse=True
        )

        if mode == "simple":
            # 3a. No clustering — keep every back-projected point as one
            # object, no noise rejection or instance splitting.
            fused = ProjectedObject(
                id="fused",
                path="",
                points=ds_points,
                colors=ds_colors,
                bbox=robust_bounds(ds_points, *self.bbox_percentile),
            )
            return state.model_copy(update={"projected": [fused]})

        if mode == "cluster_single":
            # 3b. One fused object: dominant cluster + tight world box.
            mask = largest_cluster(
                ds_points, self.cluster_eps, self.cluster_min_samples
            )
            obj_points = ds_points[mask]
            obj_colors = ds_colors[mask] if ds_colors is not None else None
            fused = ProjectedObject(
                id="fused",
                path="",
                points=obj_points,
                colors=obj_colors,
                bbox=robust_bounds(obj_points, *self.bbox_percentile),
            )
            return state.model_copy(update={"projected": [fused]})

        # 3c. cluster_instances: one object per cluster, most multi-frame
        # consensus first.
        masks = cluster_masks(
            ds_points,
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
            min_size=self.min_instance_size,
        )
        if not masks:
            # All points labelled noise (or every cluster below the size
            # floor) — keep everything as one object so the caller still
            # gets a usable box, mirroring largest_cluster's fallback.
            masks = [np.ones(len(ds_points), dtype=bool)]
        if self.min_views > 1:
            # Multi-frame consensus: a real object is segmented from several
            # viewpoints, a SAM false positive usually from just one. `m` is a
            # mask over kept points; m[inverse] lifts it back to the original
            # points, whose frame ids we can then count.
            kept = [
                m for m in masks
                if np.unique(all_frames[m[inverse]]).size >= self.min_views
            ]
            if not kept:
                warnings.warn(
                    f"ProjectTo3D: every cluster was backed by fewer than "
                    f"min_views={self.min_views} frames — nothing survived the "
                    "consensus filter. Lower min_views if the object is only "
                    "visible in a few frames.",
                    stacklevel=2,
                )
            masks = kept
        if self.max_instances is not None:
            masks = masks[: self.max_instances]

        projected = [
            ProjectedObject(
                id=f"obj-{i}",
                path="",
                points=ds_points[m],
                colors=ds_colors[m] if ds_colors is not None else None,
                bbox=robust_bounds(ds_points[m], *self.bbox_percentile),
            )
            for i, m in enumerate(masks)
        ]
        return state.model_copy(update={"projected": projected})
