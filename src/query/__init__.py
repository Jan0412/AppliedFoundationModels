"""2D image search pipeline (text → SigLIP → LanceDB → detector rerank).

Usage::

    from src.query import Search2D, SearchState

    pipeline = Search2D.from_config("config.yaml", detector="sam")
    state = pipeline.invoke(query="keyboard",
                            collection_id="rgbd_dataset_freiburg1_desk",
                            top_k_retrieve=20,
                            top_k_final=5)

    for hit in state.results:
        print(hit.path, hit.similarity_score, hit.detection_score)

Every step is a ``Runnable[SearchState, SearchState]``, so partial chains
work out of the box::

    chain = pipeline.embed | pipeline.retrieve | pipeline.detect
"""

from src.data_model import (
    DetectedImage,
    ProjectedObject,
    RetrievedImage,
    SearchState,
)

from .detect import Detect
from .embed import EmbedQuery
from .pipeline import Search2D
from .project import ProjectTo3D
from .rerank import RerankByDetection
from .retrieve import RetrieveSimilar

__all__ = [
    "Search2D",
    "SearchState",
    "RetrievedImage",
    "DetectedImage",
    "ProjectedObject",
    "EmbedQuery",
    "RetrieveSimilar",
    "Detect",
    "RerankByDetection",
    "ProjectTo3D",
]
