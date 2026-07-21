"""Scene reconstruction: RGB frames → depth + poses + intrinsics.

:class:`VGGTOmegaReconstructor` is the default backend
(``reconstruct.backend: vggt_omega`` in ``config.yaml``, selected by
:class:`~src.ingest.VideoIngestor.from_config`).
:class:`MockVGGTReconstructor` is the ScanNet GT passthrough
(``reconstruct.backend: mock``). See :mod:`src.reconstruct.base` for the
shared contract.
"""

from .base import BaseReconstructor
from .mock_vggt import MockVGGTReconstructor
from .vggt_omega import VGGTOmegaReconstructor

__all__ = ["BaseReconstructor", "MockVGGTReconstructor", "VGGTOmegaReconstructor"]
