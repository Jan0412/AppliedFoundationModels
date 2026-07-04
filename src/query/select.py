"""Step 3 — trim the retrieval pool to n viewpoint-diverse frames.

The retrieval pool is intentionally generous (hundreds of frames) and highly
redundant — consecutive video frames show the object from nearly the same
pose. This step keeps the detector budget fixed while maximising *viewpoint*
coverage of the object:

1. Per frame, estimate the world point the camera looks at
   (:func:`~src.utils.geometry.lookat_point` — camera centre + optical axis
   × robust central depth).
2. The component-wise **median** of those look-at points is a coarse object
   position ``o`` (robust to frames whose centre pixel misses the object).
3. Each frame's viewing direction ``v = normalize(cam_center - o)`` says
   *from which side* it sees the object — independent of distance.
4. Greedy farthest-point sampling on the angle between viewing directions
   picks ``n`` frames spread around the object, seeded with the
   highest-similarity frame so the best hit always reaches the detector.

Only the selected frames get their images loaded.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
from langchain_core.runnables import Runnable, RunnableConfig

from src.data_model import RetrievalDiagnostics, RetrievedImage, SearchState
from src.utils.db import load_collection_meta
from src.utils.geometry import central_depth, farthest_point_sampling, lookat_point


class SelectDiverse(Runnable):
    """Keep the ``n_diverse`` most viewpoint-diverse frames of the pool.

    Selection is **pure diversity**: similarity already formed the pool (and
    seeds the selection), so no combined relevance/diversity score is used.

    Falls back to the top-``n`` frames by similarity when the geometry is
    unusable (no ``_collection_meta`` calibration, or no frame has a valid
    depth + pose). Frames without valid geometry can still be chosen as
    similarity-ordered fill when fewer than ``n`` valid frames exist.

    When the pool was built in ``"topk"`` mode the step is a pass-through —
    the legacy fixed-k pipeline behaves exactly as before.

    Pre:  ``state.retrieved`` is set (similarity-desc, images may be unloaded).
    Post: ``state.retrieved`` holds ``<= n`` frames (similarity-desc), each
          with its image loaded; ``state.retrieval_diag`` gains
          ``n_selected`` + ``selected_ids``.
    """

    def __init__(self, db, n_diverse: int = 10, patch_frac: float = 0.2) -> None:
        self.db = db
        self.n_diverse = n_diverse
        self.patch_frac = patch_frac
        self._meta_cache: dict[str, Optional[dict]] = {}

    def _depth_scale(self, collection_id: str) -> Optional[float]:
        if collection_id not in self._meta_cache:
            self._meta_cache[collection_id] = load_collection_meta(
                self.db, collection_id
            )
        meta = self._meta_cache[collection_id]
        return None if meta is None else float(meta["depth_scale"])

    @staticmethod
    def _lookat(frame: RetrievedImage, depth_scale: float, patch_frac: float) -> Optional[np.ndarray]:
        """Look-at point of *frame*, or ``None`` when its geometry is unusable."""
        if not frame.depth_path or frame.cam2world is None:
            return None
        try:
            depth_m = central_depth(frame.depth_path, depth_scale, patch_frac)
        except OSError as exc:
            warnings.warn(
                f"SelectDiverse: cannot read depth for '{frame.id}' ({exc}).",
                stacklevel=2,
            )
            return None
        if depth_m is None:
            return None
        return lookat_point(frame.cam2world, depth_m)

    def _finish(
        self,
        state: SearchState,
        selection: list[RetrievedImage],
    ) -> SearchState:
        """Sort, load images, and write selection + diagnostics back."""
        selection = sorted(
            selection, key=lambda f: f.similarity_score, reverse=True
        )
        for frame in selection:
            frame.load_image()
        diag = state.retrieval_diag or RetrievalDiagnostics(
            mode="dynamic", pool_size=len(state.retrieved or [])
        )
        diag = diag.model_copy(
            update={
                "n_selected": len(selection),
                "selected_ids": [f.id for f in selection],
            }
        )
        return state.model_copy(
            update={"retrieved": selection, "retrieval_diag": diag}
        )

    def invoke(
        self,
        state: SearchState,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> SearchState:
        if state.retrieved is None:
            raise ValueError(
                "SelectDiverse: state.retrieved is None — run RetrieveSimilar first."
            )
        if state.retrieval_diag is not None and state.retrieval_diag.mode == "topk":
            return state  # legacy fixed-k path: nothing to trim

        pool = state.retrieved
        n = state.n_diverse or self.n_diverse
        if len(pool) <= n:
            return self._finish(state, list(pool))

        depth_scale = self._depth_scale(state.collection_id)
        if depth_scale is None:
            warnings.warn(
                f"SelectDiverse: no calibration for collection "
                f"'{state.collection_id}' — falling back to top-{n} by "
                "similarity.",
                stacklevel=2,
            )
            return self._finish(state, list(pool[:n]))

        lookats = [
            self._lookat(frame, depth_scale, self.patch_frac) for frame in pool
        ]
        valid_idx = [i for i, la in enumerate(lookats) if la is not None]
        if not valid_idx:
            warnings.warn(
                "SelectDiverse: no frame has a valid depth + pose — falling "
                f"back to top-{n} by similarity.",
                stacklevel=2,
            )
            return self._finish(state, list(pool[:n]))

        # Coarse object position: median of the look-at points.
        obj_center = np.median(
            np.stack([lookats[i] for i in valid_idx]), axis=0
        )

        # Viewing directions object → camera; drop degenerate frames whose
        # camera centre coincides with the object estimate.
        directions: list[np.ndarray] = []
        dir_idx: list[int] = []
        for i in valid_idx:
            offset = pool[i].cam2world[:3, 3] - obj_center
            norm = float(np.linalg.norm(offset))
            if norm < 1e-6:
                continue
            directions.append(offset / norm)
            dir_idx.append(i)

        if not dir_idx:
            warnings.warn(
                "SelectDiverse: all viewing directions degenerate — falling "
                f"back to top-{n} by similarity.",
                stacklevel=2,
            )
            return self._finish(state, list(pool[:n]))

        # Seed with the highest-similarity usable frame (pool is sim-desc),
        # then maximise angular spread around the object.
        picked_local = farthest_point_sampling(
            np.stack(directions), seed_idx=0, n=min(n, len(dir_idx))
        )
        picked = {dir_idx[j] for j in picked_local}

        # Fewer usable frames than n: fill with the remaining frames in
        # similarity order.
        if len(picked) < n:
            for i in range(len(pool)):
                if len(picked) >= n:
                    break
                picked.add(i)

        return self._finish(state, [pool[i] for i in sorted(picked)])
