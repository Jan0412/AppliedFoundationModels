"""Step 2 — retrieve top-k similar images from LanceDB."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import Runnable, RunnableConfig
from PIL import Image

from src.data_model import RetrievedImage, SearchState


class RetrieveSimilar(Runnable):
    """Search the LanceDB table for the ``top_k_retrieve`` nearest vectors.

    The class does **not** open a connection itself; the caller passes in
    an already-open :class:`lancedb.DBConnection` (obtain it via
    :func:`src.utils.db.connect`).

    Pre:  ``state.query_embedding`` is set; a table named
          ``state.collection_id`` exists in the connected DB.
    Post: ``state.retrieved`` is a list of :class:`RetrievedImage`,
          length ``<= state.top_k_retrieve``, each PIL image eagerly loaded.
    """

    def __init__(self, db) -> None:
        self.db = db

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

        table = self.db.open_table(state.collection_id)
        # Vectors are stored L2-normalised (see SigLIPModel). LanceDB's
        # default L2 metric returns the *squared* distance in _distance,
        # so cosine similarity = 1 - _distance / 2.
        rows = (
            table.search(state.query_embedding.tolist())
            .limit(state.top_k_retrieve)
            .to_list()
        )

        retrieved: list[RetrievedImage] = []
        for row in rows:
            dist_sq = float(row["_distance"])
            similarity = 1.0 - dist_sq / 2.0
            path = row["path"]
            image = Image.open(path).convert("RGB")
            retrieved.append(
                RetrievedImage(
                    id=row["id"],
                    path=path,
                    similarity_score=similarity,
                    image=image,
                )
            )

        return state.model_copy(update={"retrieved": retrieved})
