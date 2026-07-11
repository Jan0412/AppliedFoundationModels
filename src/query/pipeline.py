"""2D image search pipeline: text query → SigLIP → LanceDB → detector rerank."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import yaml

from src.data_model import SearchState
from src.models import SAMModel, SigLIPModel
from src.utils.db import connect as _db_connect

from .detect import Detect
from .embed import EmbedQuery
from .project import ProjectTo3D
from .rerank import RerankByDetection
from .retrieve import RetrieveSimilar
from .select import SelectDiverse


class Search2D:
    """Composes the six pipeline steps into a single LCEL chain.

    ``embed | retrieve | select | detect | rerank | project``

    Retrieval runs in one of two modes (``retrieval_mode``, per-query
    overridable via ``invoke(retrieval_mode=...)``):

    - ``"dynamic"`` (default) — :class:`RetrieveSimilar` scores every frame
      and sizes the pool from the similarity histogram (Otsu threshold +
      separability gate); :class:`SelectDiverse` then trims the pool to
      ``n_diverse`` viewpoint-diverse frames before the detector runs.
    - ``"topk"`` — legacy fixed-k retrieval (``top_k_retrieve`` frames);
      :class:`SelectDiverse` passes through untouched.

    The step instances are exposed as public attributes (:attr:`embed`,
    :attr:`retrieve`, :attr:`select`, :attr:`detect`, :attr:`rerank`,
    :attr:`project`) so callers can build partial chains. For example, to
    stop at 2D results and skip 3D projection::

        chain = pipeline.embed | pipeline.retrieve | pipeline.select \\
                | pipeline.detect | pipeline.rerank
        state = chain.invoke(SearchState(query="...", collection_id="..."))
        # state.results populated, state.projected stays None.

    The final :attr:`project` step back-projects each result's SAM mask into
    the 3D world point cloud (``state.projected``); it requires the SAM
    detector and a collection indexed with depth + poses + calibration. Its
    ``mode`` (``"simple"`` no clustering | ``"cluster_single"`` one fused
    object — default — | ``"cluster_instances"`` one object per spatial
    cluster, most consensus first) is likewise overridable per query via
    ``invoke(mode=...)``.

    The detector is :class:`SAMModel` (SAM3), exposing the
    ``invoke({"image": pil, "text": str}) -> dict`` contract. Internally
    :class:`Detect` is duck-typed, so any wrapper that satisfies the same
    contract can still be passed directly to the constructor.

    Example::

        pipeline = Search2D.from_config("config.yaml")
        state = pipeline.invoke(query="laptop on desk",
                                collection_id="fr1_desk",
                                top_k_final=5)
        for hit in state.results:
            print(hit.path, hit.similarity_score, hit.detection_score)
        print(state.retrieval_diag)   # pool size, threshold, η, ...
    """

    def __init__(
        self,
        siglip: SigLIPModel,
        detector: SAMModel,
        db,
        *,
        retrieval_mode: str = "dynamic",
        min_k: int = 10,
        max_k: int = 400,
        min_separability: float = 0.75,
        strategy: str = "tail",
        n_diverse: int = 10,
        patch_frac: float = 0.2,
        mode: str = "cluster_single",
        max_instances: Optional[int] = None,
        min_instance_size: int = 50,
        voxel: float = 0.02,
        cluster_eps: float = 0.05,
        cluster_min_samples: int = 10,
        bbox_percentile: Tuple[float, float] = (2.0, 98.0),
    ) -> None:
        self.embed = EmbedQuery(siglip)
        self.retrieve = RetrieveSimilar(
            db,
            mode=retrieval_mode,
            min_k=min_k,
            max_k=max_k,
            min_separability=min_separability,
            strategy=strategy,
        )
        self.select = SelectDiverse(db, n_diverse=n_diverse, patch_frac=patch_frac)
        self.detect = Detect(detector)
        self.rerank = RerankByDetection()
        self.project = ProjectTo3D(
            db,
            voxel=voxel,
            mode=mode,
            cluster_eps=cluster_eps,
            cluster_min_samples=cluster_min_samples,
            bbox_percentile=bbox_percentile,
            max_instances=max_instances,
            min_instance_size=min_instance_size,
        )
        self.chain = (
            self.embed | self.retrieve | self.select
            | self.detect | self.rerank | self.project
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
            detector: Which detector to wire — only ``"sam"`` (SAM3) is
                      supported; any other value raises ``ValueError``.

        Reads ``indexing.db_path`` for the LanceDB store, the optional
        ``query`` section for retrieval/selection parameters, and the optional
        ``projection`` section for 3D fuse parameters (defaults apply when a
        section is absent); SAM and SigLIP load their own sections via their
        respective ``from_config`` classmethods.
        """
        cfg = yaml.safe_load(Path(path).read_text())
        siglip = SigLIPModel.from_config(path)
        if detector != "sam":
            raise ValueError(
                f"Search2D.from_config: unknown detector {detector!r}. "
                "Expected 'sam'."
            )
        det = SAMModel.from_config(path)
        db = _db_connect(cfg["indexing"]["db_path"])
        qcfg = cfg.get("query") or {}
        pcfg = cfg.get("projection") or {}
        return cls(
            siglip=siglip,
            detector=det,
            db=db,
            retrieval_mode=qcfg.get("retrieval_mode", "dynamic"),
            min_k=qcfg.get("min_k", 10),
            max_k=qcfg.get("max_k", 400),
            min_separability=qcfg.get("min_separability", 0.75),
            strategy=qcfg.get("strategy", "tail"),
            n_diverse=qcfg.get("n_diverse", 10),
            patch_frac=qcfg.get("patch_frac", 0.2),
            mode=pcfg.get("mode", "cluster_single"),
            max_instances=pcfg.get("max_instances"),
            min_instance_size=pcfg.get("min_instance_size", 50),
            voxel=pcfg.get("voxel", 0.02),
            cluster_eps=pcfg.get("cluster_eps", 0.05),
            cluster_min_samples=pcfg.get("cluster_min_samples", 10),
            bbox_percentile=tuple(pcfg.get("bbox_percentile", (2.0, 98.0))),
        )

    def invoke(
        self,
        state: Optional[SearchState] = None,
        *,
        query: Optional[str] = None,
        collection_id: Optional[str] = None,
        top_k_retrieve: int = 20,
        top_k_final: int = 5,
        retrieval_mode: Optional[str] = None,
        n_diverse: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> SearchState:
        """Run the full chain.

        Accepts either a pre-built :class:`SearchState` or the constructor
        kwargs. ``retrieval_mode`` / ``n_diverse`` / ``mode`` override the
        configured step defaults for this query only (``top_k_retrieve``
        applies in ``"topk"`` mode only). Returns the final state with
        ``results`` set.
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
                retrieval_mode=retrieval_mode,
                n_diverse=n_diverse,
                mode=mode,
            )
        return self.chain.invoke(state)
