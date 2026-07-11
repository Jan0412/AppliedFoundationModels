"""Disk cache for scene point clouds.

Building a scene cloud back-projects a few hundred depth maps —
seconds of work, and the same result every time. The UI needs it on every scene
switch, so it is cached as an ``.npz`` next to the store and rebuilt only when
the inputs that shaped it change.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.utils.datasets import load_collection
from src.utils.geometry import build_scene_cloud

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class SceneCloudCache:
    """Cache of built scene clouds, keyed by collection id.

    Args:
        cache_dir: Directory holding the ``.npz`` files (created on write).
        n_frames:  Frames back-projected into a cloud.
        voxel:     Downsample edge (m).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        n_frames: int = 250,
        voxel: float = 0.02,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.n_frames = n_frames
        self.voxel = voxel

    def path_for(self, collection_id: str) -> Path:
        """Cache file for *collection_id*.

        Raises:
            ValueError: If the id could escape the cache dir or collide as a
                filename — ids reach here from user-named uploads.
        """
        if not _SAFE_ID.match(collection_id):
            raise ValueError(
                f"unsafe collection id for a cache filename: {collection_id!r}"
            )
        return self.cache_dir / f"{collection_id}.npz"

    def get(
        self,
        db,
        collection_id: str,
        *,
        force: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(points, colors)`` for *collection_id*, building it if needed.

        A cached cloud is only reused when it was built with the current
        ``n_frames``/``voxel`` and from the same number of rows — so re-indexing
        or re-ingesting a collection invalidates it without anyone remembering to.

        Args:
            db:            An open :class:`lancedb.DBConnection`.
            collection_id: Collection to render.
            force:         Rebuild even on a valid hit.
        """
        row_count = db.open_table(collection_id).count_rows()
        path = self.path_for(collection_id)

        if not force and path.is_file():
            cached = self._read(path, row_count)
            if cached is not None:
                return cached

        frames = load_collection(db, collection_id)
        points, colors = build_scene_cloud(
            frames, n_frames=self.n_frames, voxel=self.voxel
        )
        self._write(path, points, colors, row_count)
        return points, colors

    def invalidate(self, collection_id: str) -> None:
        """Drop the cached cloud for *collection_id*, if any."""
        self.path_for(collection_id).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # npz round-trip
    # ------------------------------------------------------------------

    def _read(
        self, path: Path, row_count: int
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Load a cached cloud, or ``None`` if it is stale or unreadable."""
        try:
            with np.load(path) as data:
                if (
                    int(data["n_frames"]) != self.n_frames
                    or float(data["voxel"]) != self.voxel
                    or int(data["row_count"]) != row_count
                ):
                    return None
                colors = data["colors"] if bool(data["has_colors"]) else None
                return data["points"], colors
        except (OSError, KeyError, ValueError):
            # A truncated or older-format file is not worth failing a page load
            # over — treat it as a miss and rebuild.
            return None

    def _write(
        self,
        path: Path,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        row_count: int,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            points=points,
            colors=colors if colors is not None else np.empty((0, 3), dtype=np.uint8),
            has_colors=colors is not None,
            n_frames=self.n_frames,
            voxel=self.voxel,
            row_count=row_count,
        )
