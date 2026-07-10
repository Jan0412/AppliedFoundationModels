"""Step 2 — retrieve similar images from LanceDB (fixed top-k or dynamic pool)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from langchain_core.runnables import Runnable, RunnableConfig

from src.data_model import RetrievalDiagnostics, RetrievedImage, SearchState

from .dynamic_k import dynamic_pool_size

#: Columns materialised per hit. The stored embedding (``vector``) is never
#: fetched — it is only needed inside LanceDB's distance computation, and in
#: dynamic mode the query returns *every* row of the collection. ``_distance``
#: must be requested explicitly once output columns are constrained.
_COLUMNS = ["id", "path", "depth_path", "cam2world", "_distance"]


class RetrieveSimilar(Runnable):
    """Search the LanceDB table for similar frames.

    Two modes (constructor default, overridable per query via
    ``state.retrieval_mode``):

    - ``"topk"``    — legacy fixed-size retrieval: the ``top_k_retrieve``
      nearest vectors, no thresholding.
    - ``"dynamic"`` — scores **all** frames in the collection and sizes the
      pool from the similarity histogram (:func:`dynamic_pool_size`): Otsu
      threshold (``strategy``), separability gate (``min_separability`` —
      below it the object is likely absent and the pool falls back to
      ``min_k``), clamped to ``[min_k, max_k]``. ``state.top_k_retrieve``
      is ignored in this mode.

    Images are **not** loaded in either mode — the dynamic pool can span
    hundreds of frames. Downstream steps call
    :meth:`~src.data_model.RetrievedImage.load_image` on the frames they
    actually consume.

    The class does **not** open a connection itself; the caller passes in
    an already-open :class:`lancedb.DBConnection` (obtain it via
    :func:`src.utils.db.connect`).

    Pre:  ``state.query_embedding`` is set; a table named
          ``state.collection_id`` exists in the connected DB.
    Post: ``state.retrieved`` is a list of :class:`RetrievedImage` sorted by
          similarity descending (images ``None``); ``state.retrieval_diag``
          records how the pool was formed.
    """

    def __init__(
        self,
        db,
        *,
        mode: str = "topk",
        min_k: int = 10,
        max_k: int = 400,
        min_separability: float = 0.75,
        strategy: str = "tail",
    ) -> None:
        if mode not in ("topk", "dynamic"):
            raise ValueError(
                f"RetrieveSimilar: unknown mode {mode!r}. "
                "Expected 'topk' or 'dynamic'."
            )
        if strategy not in ("tail", "plain"):
            raise ValueError(
                f"RetrieveSimilar: unknown strategy {strategy!r}. "
                "Expected 'tail' or 'plain'."
            )
        self.db = db
        self.mode = mode
        self.min_k = min_k
        self.max_k = max_k
        self.min_separability = min_separability
        self.strategy = strategy

    @staticmethod
    def _to_retrieved(row: dict) -> RetrievedImage:
        """Build a :class:`RetrievedImage` (image deferred) from a LanceDB row."""
        # Searched with LanceDB's cosine metric (see ``distance_type`` on the
        # queries below), so _distance holds the cosine *distance*
        # (1 - cosine similarity). Convert back to similarity so downstream
        # steps keep their "higher = closer" contract.
        similarity = 1.0 - float(row["_distance"])
        cam = row.get("cam2world")
        cam2world = (
            np.asarray(cam, dtype=np.float32).reshape(4, 4)
            if cam is not None
            else None
        )
        return RetrievedImage(
            id=row["id"],
            path=row["path"],
            similarity_score=similarity,
            image=None,
            depth_path=row.get("depth_path", "") or "",
            cam2world=cam2world,
        )

    def invoke(
        self,
        state: SearchState,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> SearchState:
        if state.query_embedding is None:
            raise ValueError(
                "RetrieveSimilar: state.query_embedding is None — run EmbedQuery first."
            )

        mode = state.retrieval_mode or self.mode
        table = self.db.open_table(state.collection_id)

        if mode == "topk":
            rows = (
                table.search(state.query_embedding.tolist())
                .distance_type("cosine")
                .select(_COLUMNS)
                .limit(state.top_k_retrieve)
                .to_list()
            )
            retrieved = [self._to_retrieved(row) for row in rows]
            diag = RetrievalDiagnostics(mode="topk", pool_size=len(retrieved))
            return state.model_copy(
                update={"retrieved": retrieved, "retrieval_diag": diag}
            )

        # ── dynamic mode ─────────────────────────────────────────────────────
        n_total = table.count_rows()
        if n_total == 0:
            diag = RetrievalDiagnostics(
                mode="dynamic", strategy=self.strategy, gated=True,
                pool_size=0, total_frames=0,
            )
            return state.model_copy(
                update={"retrieved": [], "retrieval_diag": diag}
            )

        # Rows arrive sorted by _distance ascending = similarity descending,
        # so the thresholded pool is simply the first `cut.k` rows.
        rows = (
            table.search(state.query_embedding.tolist())
            .distance_type("cosine")
            .select(_COLUMNS)
            .limit(n_total)
            .to_list()
        )
        sims = np.array(
            [1.0 - float(r["_distance"]) for r in rows], dtype=np.float32
        )
        cut = dynamic_pool_size(
            sims,
            strategy=self.strategy,
            min_k=self.min_k,
            max_k=self.max_k,
            min_separability=self.min_separability,
        )

        retrieved = [self._to_retrieved(row) for row in rows[: cut.k]]
        diag = RetrievalDiagnostics(
            mode="dynamic",
            strategy=self.strategy,
            threshold=cut.threshold,
            separability=cut.separability,
            gated=cut.gated,
            pool_size=len(retrieved),
            total_frames=n_total,
        )
        return state.model_copy(
            update={"retrieved": retrieved, "retrieval_diag": diag}
        )
