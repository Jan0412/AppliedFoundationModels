"""Pick the embedding model — and its LanceDB store — from config.

The rest of the pipeline is embedder-agnostic: :class:`~src.index.Indexer` and
:class:`~src.query.pipeline.Search2D` only ever call ``embed_images`` /
``embed_text`` and read ``embedding_dim``. This module is the single place that
decides *which* wrapper that is, so swapping SigLIP for CLIP is one config key::

    models:
      embedder: "clip"

or, for a one-off A/B run that leaves the file alone::

    AFM_EMBEDDER=clip uv run python scripts/eval_scanrefer_unique.py

The store lives here too, deliberately. Vectors from two embedders share
neither dimension nor space, so a collection indexed with one is unusable by
the other — the embedder choice and the LanceDB path have to move together, and
:func:`db_path_for` keeps that invariant next to the choice that creates it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from .base import BaseModel
from .clip import CLIPEmbedModel
from .siglib import SigLIPModel

#: Embedder name (the ``models.<name>`` config section) → wrapper class.
EMBEDDERS: dict[str, type[BaseModel]] = {
    "siglip": SigLIPModel,
    "clip": CLIPEmbedModel,
}

#: Used when neither the env override nor ``models.embedder`` says otherwise.
DEFAULT_EMBEDDER = "siglip"

#: Env var that overrides ``models.embedder`` for a single run.
ENV_VAR = "AFM_EMBEDDER"


def embedder_name(cfg: dict, override: Optional[str] = None) -> str:
    """Resolve which embedder to use, most specific source first.

    Precedence: *override* argument → ``$AFM_EMBEDDER`` → ``models.embedder``
    in *cfg* → :data:`DEFAULT_EMBEDDER`.

    Args:
        cfg:      Parsed config mapping.
        override: Explicit name that beats both config and environment.

    Raises:
        ValueError: If the resolved name is not in :data:`EMBEDDERS`.
    """
    name = (
        override
        or os.environ.get(ENV_VAR)
        or (cfg.get("models") or {}).get("embedder")
        or DEFAULT_EMBEDDER
    )
    if name not in EMBEDDERS:
        raise ValueError(
            f"unknown embedder {name!r} — expected one of "
            f"{', '.join(sorted(EMBEDDERS))}."
        )
    return name


def load_embedder(
    path: str | Path = "config.yaml",
    name: Optional[str] = None,
) -> BaseModel:
    """Build the configured embedder from *path* (a YAML config file).

    The chosen wrapper reads its own ``models.<name>`` section via
    :meth:`~src.models.base.BaseModel.from_config`, so each embedder keeps its
    own ``model_id`` / ``device`` / ``batch_size``.

    Args:
        path: Path to the YAML configuration file.
        name: Explicit embedder name; see :func:`embedder_name` for precedence.

    Example::

        model = load_embedder("config.yaml")          # honours the config
        model = load_embedder("config.yaml", "clip")  # forces CLIP
    """
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    return EMBEDDERS[embedder_name(cfg, name)].from_config(path)


def db_path_for(cfg: dict, name: Optional[str] = None) -> str:
    """Return the LanceDB store belonging to the configured embedder.

    ``indexing.db_paths`` maps embedder name → store directory; a missing
    entry (or a missing mapping) falls back to ``indexing.db_path``. Keeping
    the stores apart is not optional: LanceDB fixes the vector column's width
    when the table is created, so writing 512-d CLIP rows into a 768-d SigLIP
    table fails, and a query embedded by the wrong model would be meaningless
    even if the widths happened to match.

    Args:
        cfg:  Parsed config mapping.
        name: Explicit embedder name; see :func:`embedder_name` for precedence.

    Raises:
        KeyError: If neither ``indexing.db_paths[<name>]`` nor
            ``indexing.db_path`` is set.
    """
    idx_cfg = cfg.get("indexing") or {}
    per_embedder = idx_cfg.get("db_paths") or {}
    path = per_embedder.get(embedder_name(cfg, name)) or idx_cfg.get("db_path")
    if not path:
        raise KeyError(
            "config is missing a LanceDB store: set indexing.db_path, or "
            "indexing.db_paths.<embedder>."
        )
    return str(path)
