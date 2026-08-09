"""Step 1 — embed the text query with the configured embedder (SigLIP2 or CLIP)."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import Runnable, RunnableConfig

from src.data_model import SearchState
from src.models import BaseModel


class EmbedQuery(Runnable):
    """Embed ``state.query`` with the embedder and store it on the state.

    The embedder must be the same model the collection was indexed with —
    :func:`~src.models.factory.load_embedder` and
    :func:`~src.models.factory.db_path_for` keep those two in step.

    Pre:  ``state.query`` is a non-empty string.
    Post: ``state.query_embedding`` is a 1-D L2-normalised :class:`np.ndarray`.
    """

    def __init__(self, embedder: BaseModel) -> None:
        self.embedder = embedder

    def invoke(
        self,
        state: SearchState,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> SearchState:
        if not state.query:
            raise ValueError("EmbedQuery: state.query is empty.")
        embedding = self.embedder.embed_text(state.query)
        return state.model_copy(update={"query_embedding": embedding})
