"""LCEL-compatible HuggingFace model wrappers.

Usage::

    from src.models import SigLIPModel, SAMModel, SamViTModel, GroundingDINOModel

    sig     = SigLIPModel.from_config("config.yaml")
    sam     = SAMModel.from_config("config.yaml")
    sam_vit = SamViTModel.from_config("config.yaml")
    dino    = GroundingDINOModel.from_config("config.yaml")
"""

from .base import BaseModel
from .grounding_dino import GroundingDINOModel
from .sam import SAMModel
from .sam_vit import SamViTModel
from .siglib import SigLIPModel

__all__ = ["BaseModel", "SigLIPModel", "SAMModel", "SamViTModel", "GroundingDINOModel"]
