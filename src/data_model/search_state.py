"""Pydantic state object that flows through the 2D image search chain.

A single :class:`SearchState` is the **only** input/output type for every
step in :mod:`src.query`. Each step reads a subset of its fields and
writes a different subset; nothing else crosses the boundary. This is
what makes the chain stackable — any ``Runnable[SearchState, SearchState]``
slots in.

Fields populated incrementally:

- :class:`EmbedQuery`         writes ``query_embedding``
- :class:`RetrieveSimilar`    writes ``retrieved``
- :class:`Detect`             writes ``detected``
- :class:`RerankByDetection`  writes ``results``

Unused fields stay ``None`` after a partial chain, so callers can branch
on which fields are populated without dealing with half-built objects.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict


class RetrievedImage(BaseModel):
    """One LanceDB hit, after :class:`RetrieveSimilar`.

    Attributes:
        id:               LanceDB row id (matches the indexer's id scheme).
        path:             Source image path on disk (may be empty if the
                          original was a PIL image).
        similarity_score: Cosine similarity to the query embedding (high = closer).
        image:            Lazily-loaded RGB :class:`PIL.Image.Image`.
    """

    id: str
    path: str
    similarity_score: float
    image: Image.Image

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DetectedImage(BaseModel):
    """One candidate after a detector (SAM, Grounding DINO, …) has run on it.

    The scalar fields (``id``, ``path``, ``similarity_score``) are carried
    over from the upstream :class:`RetrievedImage` so the rerank step never
    has to look back at ``state.retrieved`` to recover provenance.

    ``masks`` and ``labels`` are detector-dependent: SAM produces ``masks``,
    Grounding DINO produces ``labels``. Whichever the chosen detector did
    not return is left ``None`` — downstream code branches with
    ``if img.masks is not None: ...``.

    Attributes:
        id:                LanceDB row id.
        path:              Source image path.
        similarity_score:  Cosine similarity from retrieval.
        detection_score:   Per-image rank metric = ``float(scores.max())``,
                           or ``0.0`` if the detector returned no items.
        boxes:             ``(N, 4)`` float tensor of bounding boxes.
        scores:            Full detector confidence tensor.
        masks:             SAM-only: list of boolean tensors, one per segment.
        labels:            Grounding-DINO-only: list of matched label strings.
    """

    id: str
    path: str
    similarity_score: float
    detection_score: float
    boxes: torch.Tensor
    scores: torch.Tensor
    masks: Optional[list[Any]] = None
    labels: Optional[list[str]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SearchState(BaseModel):
    """The single I/O type for every step in the 2D search chain.

    Inputs (caller-supplied) define what to search for; optional fields
    are populated by the chain as it progresses.

    Attributes:
        query:           The text prompt to search for.
        collection_id:   LanceDB table name to search.
        top_k_retrieve:  Candidate pool size from LanceDB (>= top_k_final).
        top_k_final:     Final number of results after re-ranking.

        query_embedding: Set by :class:`EmbedQuery`. 1-D, L2-normalised.
        retrieved:       Set by :class:`RetrieveSimilar`.
        detected:        Set by :class:`Detect` (full, unsorted).
        results:         Set by :class:`RerankByDetection`
                         (sorted desc by ``detection_score``, trimmed).
    """

    query: str
    collection_id: str
    top_k_retrieve: int = 20
    top_k_final: int = 5

    query_embedding: Optional[np.ndarray] = None
    retrieved: Optional[list[RetrievedImage]] = None
    detected: Optional[list[DetectedImage]] = None
    results: Optional[list[DetectedImage]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
