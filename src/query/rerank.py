"""Step 4 — re-rank detected candidates and keep ``top_k_final``."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import Runnable, RunnableConfig

from src.data_model import SearchState


class RerankByDetection(Runnable):
    """Pure ordering: sort by ``detection_score`` desc, keep ``top_k_final``.

    No model dependency. Trivial to swap for a different scorer (e.g. a
    weighted combination of similarity and detection) — any Runnable
    obeying the ``SearchState -> SearchState`` contract slots in.

    Pre:  ``state.detected`` is set.
    Post: ``state.results`` is a sorted, trimmed copy of ``state.detected``.
    """

    def invoke(
        self,
        state: SearchState,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> SearchState:
        if state.detected is None:
            raise ValueError(
                "RerankByDetection: state.detected is None — run Detect first."
            )

        ranked = sorted(
            state.detected,
            key=lambda s: s.detection_score,
            reverse=True,
        )
        return state.model_copy(update={"results": ranked[: state.top_k_final]})
