"""2D image search pipeline: text query → SigLIP → LanceDB → detector rerank."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import yaml

from src.data_model import SearchState
from src.models import GroundingDINOModel, SAMModel, SigLIPModel
from src.utils.db import connect as _db_connect

from .detect import Detect
from .embed import EmbedQuery
from .project import ProjectTo3D
from .rerank import RerankByDetection
from .retrieve import RetrieveSimilar


Detector = Union[SAMModel, GroundingDINOModel]


class Search2D:
    """Composes the five pipeline steps into a single LCEL chain.

    The step instances are exposed as public attributes (:attr:`embed`,
    :attr:`retrieve`, :attr:`detect`, :attr:`rerank`, :attr:`project`) so
    callers can build partial chains. For example, to stop at 2D results and
    skip 3D projection::

        chain = pipeline.embed | pipeline.retrieve | pipeline.detect | pipeline.rerank
        state = chain.invoke(SearchState(query="...", collection_id="..."))
        # state.results populated, state.projected stays None.

    The final :attr:`project` step back-projects each result's SAM mask into
    the 3D world point cloud (``state.projected``); it requires the SAM
    detector and a collection indexed with depth + poses + calibration.

    The detector is one of :class:`SAMModel` or :class:`GroundingDINOModel`
    (both share the ``invoke({"image": pil, "text": str}) -> dict``
    contract). Internally :class:`Detect` is duck-typed, so any new wrapper
    that satisfies the same contract can be passed as well.

    Example::

        pipeline = Search2D.from_config("config.yaml", detector="grounding_dino")
        state = pipeline.invoke(query="laptop on desk",
                                collection_id="fr1_desk",
                                top_k_retrieve=20,
                                top_k_final=5)
        for hit in state.results:
            print(hit.path, hit.similarity_score, hit.detection_score)
    """

    def __init__(
        self,
        siglip: SigLIPModel,
        detector: Detector,
        db,
    ) -> None:
        self.embed = EmbedQuery(siglip)
        self.retrieve = RetrieveSimilar(db)
        self.detect = Detect(detector)
        self.rerank = RerankByDetection()
        self.project = ProjectTo3D(db)
        self.chain = (
            self.embed | self.retrieve | self.detect | self.rerank | self.project
        )

    @classmethod
    def from_config(
        cls,
        path: str | Path = "config.yaml",
        *,
        detector: str = "sam",
    ) -> "Search2D":
        """Build a :class:`Search2D` from a YAML config file.

        Args:
            path:     Path to the YAML configuration file.
            detector: Which detector to wire — ``"sam"`` (default) or
                      ``"grounding_dino"``. Both models read their own
                      sections of the same YAML file.

        Reads ``indexing.db_path`` for the LanceDB store; SAM, DINO, and
        SigLIP load their own sections via their respective
        ``from_config`` classmethods.
        """
        cfg = yaml.safe_load(Path(path).read_text())
        siglip = SigLIPModel.from_config(path)
        if detector == "sam":
            det: Detector = SAMModel.from_config(path)
        elif detector == "grounding_dino":
            det = GroundingDINOModel.from_config(path)
        else:
            raise ValueError(
                f"Search2D.from_config: unknown detector {detector!r}. "
                "Expected 'sam' or 'grounding_dino'."
            )
        db = _db_connect(cfg["indexing"]["db_path"])
        return cls(siglip=siglip, detector=det, db=db)

    def invoke(
        self,
        state: Optional[SearchState] = None,
        *,
        query: Optional[str] = None,
        collection_id: Optional[str] = None,
        top_k_retrieve: int = 20,
        top_k_final: int = 5,
    ) -> SearchState:
        """Run the full chain.

        Accepts either a pre-built :class:`SearchState` or the four
        constructor kwargs. Returns the final state with ``results`` set.
        """
        if state is None:
            if query is None or collection_id is None:
                raise ValueError(
                    "Search2D.invoke: provide either a SearchState or "
                    "(query, collection_id) kwargs."
                )
            state = SearchState(
                query=query,
                collection_id=collection_id,
                top_k_retrieve=top_k_retrieve,
                top_k_final=top_k_final,
            )
        return self.chain.invoke(state)
