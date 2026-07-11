"""Scene reconstruction: RGB frames → depth + poses + intrinsics.

Currently only :class:`MockVGGTReconstructor` (a ScanNet passthrough) exists;
see :mod:`src.reconstruct.base` for the contract a real VGGT backend implements.
"""

from .base import BaseReconstructor
from .mock_vggt import MockVGGTReconstructor

__all__ = ["BaseReconstructor", "MockVGGTReconstructor"]
