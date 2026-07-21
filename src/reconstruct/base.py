"""The reconstruction contract: RGB frames in, a :class:`FrameSet` out.

Everything downstream of reconstruction — :class:`~src.index.Indexer`,
:class:`~src.query.ProjectTo3D`, :func:`~src.utils.geometry.build_scene_cloud`
— already speaks one language: per-frame depth maps + 4x4 cam-to-world poses +
per-collection intrinsics + a depth scale, bundled as a
:class:`~src.utils.datasets.FrameSet`. Making that the reconstructor's output
type means a video-ingested scene is indistinguishable from a TUM or ScanNet
one, and no code past this module needs to know a reconstructor exists.

Backends
--------
:class:`~src.reconstruct.vggt_omega.VGGTOmegaReconstructor` is the production
backend: it predicts up-to-scale depth and poses, writes confidence-filtered
16-bit depth PNGs, and is the default of
:meth:`~src.ingest.VideoIngestor.from_config`
(``reconstruct.backend: vggt_omega``).
:class:`~src.reconstruct.mock_vggt.MockVGGTReconstructor` implements the same
:meth:`BaseReconstructor.reconstruct` contract but serves a fixed ScanNet
sequence; select it with ``reconstruct.backend: mock``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from src.utils.datasets import FrameSet


class BaseReconstructor(ABC):
    """Turns a list of RGB frames into a back-projectable :class:`FrameSet`."""

    @abstractmethod
    def reconstruct(
        self,
        rgb_paths: list[str],
        out_dir: str | Path,
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> FrameSet:
        """Reconstruct scene geometry for *rgb_paths*.

        Args:
            rgb_paths:   RGB frames of the scene, in capture order.
            out_dir:     Directory the reconstructor may write derived files to
                         (depth maps, poses). Implementations create it as
                         needed.
            on_progress: Called with the number of frames finished since the
                         last call, so a caller's :class:`JobStatus` can be
                         advanced without this module knowing about jobs.

        Returns:
            A :class:`FrameSet` whose ``paths`` exist on disk (retrieval
            lazy-loads frames by path later), with one depth map and one
            cam-to-world pose per frame.
        """
        raise NotImplementedError
