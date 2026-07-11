"""Stand-in for VGGT: serves a real ScanNet scene instead of reconstructing.

VGGT is not implemented yet, but the rest of the ingestion path — frame
extraction, indexing, querying, 3D projection, the UI — needs geometry to work
against. Rather than fake it with synthetic depth (which produces a shell of
points nobody can query meaningfully), this returns a *real* scene:
scene0011_00 from the ScanNet mirror, subsampled to the number of frames the
caller supplied.

So: an uploaded video is **discarded** and its scene becomes a copy of
scene0011_00. Retrieval, detection and projection then run on real frames with
real depth and real poses, which is what makes the end-to-end flow worth
demoing. When :class:`VGGTReconstructor` lands, swapping it in at
:class:`~src.ingest.VideoIngestor` is the only change — this class's output
contract is already the one the real thing must meet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from src.utils.datasets import FrameSet, load_scannet

from .base import BaseReconstructor

#: The scene every "reconstruction" resolves to until VGGT is implemented.
SCENE_DIR = "/storage/group/dataset_mirrors/scannet/scans/scene0011_00"

#: Raw-depth → metres divisor for ScanNet's 16-bit depth PNGs.
DEPTH_SCALE = 1000.0


class MockVGGTReconstructor(BaseReconstructor):
    """Return frames of a fixed ScanNet scene, ignoring the input frames.

    Args:
        scene_dir:   ScanNet sequence to serve. Defaults to :data:`SCENE_DIR`;
                     overridden in tests to point at a synthetic tree.
        depth_scale: Raw-depth → metres divisor for that scene.
    """

    def __init__(
        self,
        scene_dir: str | Path = SCENE_DIR,
        depth_scale: float = DEPTH_SCALE,
    ) -> None:
        self.scene_dir = Path(scene_dir)
        self.depth_scale = depth_scale

    def reconstruct(
        self,
        rgb_paths: list[str],
        out_dir: str | Path,
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> FrameSet:
        """Serve ``len(rgb_paths)`` frames of the ScanNet scene.

        The uploaded frames are used only to decide *how many* scene frames to
        return; *out_dir* is untouched (nothing is derived — the depth maps and
        poses already exist on disk). Frames are sampled evenly across the
        sequence so a short video still yields a scene with full spatial
        coverage rather than one corner of it.
        """
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(
                f"mock reconstruction scene not found: {self.scene_dir}. "
                "It stands in for VGGT — mount the ScanNet mirror, or point "
                "MockVGGTReconstructor at another ScanNet sequence."
            )

        scene = load_scannet(self.scene_dir, depth_scale=self.depth_scale)
        if len(scene) == 0:
            raise ValueError(f"mock reconstruction scene is empty: {self.scene_dir}")

        n = min(len(scene), len(rgb_paths))
        picks = np.linspace(0, len(scene) - 1, num=n, dtype=int)

        paths, depth_paths, poses = [], [], []
        for i in picks:
            paths.append(scene.paths[i])
            depth_paths.append(scene.depth_paths[i])
            poses.append(scene.poses[i])
            if on_progress is not None:
                on_progress(1)

        return FrameSet(
            paths=paths,
            depth_paths=depth_paths,
            poses=poses,
            intrinsics=dict(scene.intrinsics),
            depth_scale=scene.depth_scale,
        )
