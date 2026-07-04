"""Pydantic state object that flows through the 2D image search chain.

A single :class:`SearchState` is the **only** input/output type for every
step in :mod:`src.query`. Each step reads a subset of its fields and
writes a different subset; nothing else crosses the boundary. This is
what makes the chain stackable — any ``Runnable[SearchState, SearchState]``
slots in.

Fields populated incrementally:

- :class:`EmbedQuery`         writes ``query_embedding``
- :class:`RetrieveSimilar`    writes ``retrieved`` + ``retrieval_diag``
- :class:`SelectDiverse`      trims ``retrieved``, updates ``retrieval_diag``
- :class:`Detect`             writes ``detected``
- :class:`RerankByDetection`  writes ``results``

Unused fields stay ``None`` after a partial chain, so callers can branch
on which fields are populated without dealing with half-built objects.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

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
        image:            RGB :class:`PIL.Image.Image`, or ``None`` until a
                          consumer calls :meth:`load_image`. Retrieval defers
                          loading because the dynamic pool can span hundreds
                          of frames of which only a few reach the detector.
        depth_path:       Paired depth-map path stored at index time ("" if none).
        cam2world:        ``(4, 4)`` camera-to-world pose stored at index time,
                          or ``None`` when the row predates 3D indexing.
    """

    id: str
    path: str
    similarity_score: float
    image: Optional[Image.Image] = None
    depth_path: str = ""
    cam2world: Optional[np.ndarray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def load_image(self) -> Image.Image:
        """Return the RGB image, loading it from :attr:`path` on first use.

        The loaded image is cached on :attr:`image`, so repeated calls (and
        later pipeline steps sharing this instance) pay the disk read once.

        Raises:
            ValueError: When no image is cached and :attr:`path` is empty
                        (rows indexed from in-memory PIL images).
        """
        if self.image is None:
            if not self.path:
                raise ValueError(
                    f"RetrievedImage.load_image: '{self.id}' has no cached "
                    "image and no source path to load it from."
                )
            self.image = Image.open(self.path).convert("RGB")
        return self.image


class RetrievalDiagnostics(BaseModel):
    """How the retrieval pool was formed — written by :class:`RetrieveSimilar`.

    All fields are JSON-serialisable scalars so evaluation scripts can log
    them per query.

    Attributes:
        mode:         Effective retrieval mode: ``"topk"`` or ``"dynamic"``.
        strategy:     Otsu strategy used (``"tail"`` | ``"plain"``); ``None``
                      in topk mode.
        threshold:    Applied similarity threshold; ``None`` in topk mode.
        separability: Otsu η of the full score distribution; ``None`` in
                      topk mode.
        gated:        ``True`` when η fell below the gate and the pool fell
                      back to ``min_k`` (object likely absent).
        pool_size:    ``len(state.retrieved)`` right after retrieval.
        total_frames: Collection size scored in dynamic mode; ``None`` in
                      topk mode.
        n_selected:   Frames kept by :class:`SelectDiverse` (``None`` until
                      that step runs).
        selected_ids: Row ids kept by :class:`SelectDiverse`.
    """

    mode: str
    strategy: Optional[str] = None
    threshold: Optional[float] = None
    separability: Optional[float] = None
    gated: bool = False
    pool_size: int = 0
    total_frames: Optional[int] = None
    n_selected: Optional[int] = None
    selected_ids: Optional[list[str]] = None


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
        depth_path:        Paired depth-map path carried from retrieval (for 3D
                           back-projection); "" if none.
        cam2world:         ``(4, 4)`` camera-to-world pose carried from retrieval,
                           or ``None``.
    """

    id: str
    path: str
    similarity_score: float
    detection_score: float
    boxes: torch.Tensor
    scores: torch.Tensor
    masks: Optional[list[Any]] = None
    labels: Optional[list[str]] = None
    depth_path: str = ""
    cam2world: Optional[np.ndarray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProjectedObject(BaseModel):
    """One result back-projected into the 3D world point cloud.

    Produced by :class:`ProjectTo3D` from a :class:`DetectedImage`'s best mask
    plus the frame's depth map, pose, and the collection intrinsics. Holds
    everything needed to highlight the searched object inside the scene cloud.

    Attributes:
        id:     LanceDB row id of the source frame.
        path:   Source RGB image path.
        points: ``(N, 3)`` float array of world-space XYZ for the masked pixels.
        colors: ``(N, 3)`` ``uint8`` RGB sampled at those pixels, or ``None``.
        bbox:   ``(2, 3)`` axis-aligned world box ``[[min],[max]]`` over the
                points (robust-percentile trimmed), or ``None`` if no points.
    """

    id: str
    path: str
    points: np.ndarray
    colors: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SearchState(BaseModel):
    """The single I/O type for every step in the 2D search chain.

    Inputs (caller-supplied) define what to search for; optional fields
    are populated by the chain as it progresses.

    Attributes:
        query:           The text prompt to search for.
        collection_id:   LanceDB table name to search.
        top_k_retrieve:  Candidate pool size from LanceDB (>= top_k_final).
                         Only used in ``"topk"`` retrieval mode — dynamic
                         mode sizes the pool from the score distribution.
        top_k_final:     Final number of results after re-ranking.
        retrieval_mode:  Per-query override of the retrieval mode
                         (``"topk"`` fixed-k | ``"dynamic"`` Otsu pool);
                         ``None`` uses the :class:`RetrieveSimilar` default.
        n_diverse:       Per-query override of how many viewpoint-diverse
                         frames :class:`SelectDiverse` keeps; ``None`` uses
                         the step default.

        query_embedding: Set by :class:`EmbedQuery`. 1-D, L2-normalised.
        retrieved:       Set by :class:`RetrieveSimilar` (images unloaded);
                         trimmed to the diverse subset (images loaded) by
                         :class:`SelectDiverse`.
        retrieval_diag:  Set by :class:`RetrieveSimilar`, extended by
                         :class:`SelectDiverse` — how the pool was formed.
        detected:        Set by :class:`Detect` (full, unsorted).
        results:         Set by :class:`RerankByDetection`
                         (sorted desc by ``detection_score``, trimmed).
        projected:       Set by :class:`ProjectTo3D` — a **single** fused
                         :class:`ProjectedObject` (all frames' points merged,
                         clustered, and reduced to one tight 3D box), or empty.
    """

    query: str
    collection_id: str
    top_k_retrieve: int = 20
    top_k_final: int = 5
    retrieval_mode: Optional[Literal["topk", "dynamic"]] = None
    n_diverse: Optional[int] = None

    query_embedding: Optional[np.ndarray] = None
    retrieved: Optional[list[RetrievedImage]] = None
    retrieval_diag: Optional[RetrievalDiagnostics] = None
    detected: Optional[list[DetectedImage]] = None
    results: Optional[list[DetectedImage]] = None
    projected: Optional[list[ProjectedObject]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
