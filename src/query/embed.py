"""Step 1 — embed the text query with SigLIP2."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import Runnable, RunnableConfig

from src.data_model import SearchState
from src.models import SigLIPModel


class EmbedQuery(Runnable):
    """Embed ``state.query`` with SigLIP2 and store it on the state.

    Pre:  ``state.query`` is a non-empty string.
    Post: ``state.query_embedding`` is a 1-D L2-normalised :class:`np.ndarray`.
    """

    def __init__(self, siglip: SigLIPModel) -> None:
        self.siglip = siglip

    def invoke(
        self,
        state: SearchState,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> SearchState:
        if not state.query:
            raise ValueError("EmbedQuery: state.query is empty.")
        embedding = self.siglip.embed_text(state.query)
        return state.model_copy(update={"query_embedding": embedding})
