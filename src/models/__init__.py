"""LCEL-compatible HuggingFace model wrappers.

Usage::

    from src.models import load_embedder, SAMModel

    model = load_embedder("config.yaml")   # SigLIP or CLIP, per config
    sam   = SAMModel.from_config("config.yaml")

:class:`SigLIPModel` and :class:`CLIPEmbedModel` share one embedding contract
(``embed_text`` / ``embed_images`` / ``embedding_dim``); prefer
:func:`~src.models.factory.load_embedder` over naming either directly, so the
choice stays a config key.
"""

from .base import BaseModel
from .clip import CLIPEmbedModel
from .factory import EMBEDDERS, db_path_for, embedder_name, load_embedder
from .sam import SAMModel
from .siglib import SigLIPModel

__all__ = [
    "BaseModel",
    "CLIPEmbedModel",
    "EMBEDDERS",
    "SAMModel",
    "SigLIPModel",
    "db_path_for",
    "embedder_name",
    "load_embedder",
]
