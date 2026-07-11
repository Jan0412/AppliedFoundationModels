"""LCEL-compatible HuggingFace model wrappers.

Usage::

    from src.models import SigLIPModel, SAMModel

    sig = SigLIPModel.from_config("config.yaml")
    sam = SAMModel.from_config("config.yaml")
"""

from .base import BaseModel
from .sam import SAMModel
from .siglib import SigLIPModel

__all__ = ["BaseModel", "SigLIPModel", "SAMModel"]
