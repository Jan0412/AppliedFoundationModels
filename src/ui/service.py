"""The UI's view of the pipeline — everything the frontend needs, no frontend.

Deliberately free of viser (or any web framework): the viser app in
:mod:`src.ui.app` is a thin shell over this, and a future HTTP API would map
onto the same five methods (``list_scenes``, ``load_scene``, ``query``,
``ingest_video``, ``job_status``) without touching the pipeline.

Models load lazily, so the server answers immediately on start and pays for
SigLIP/SAM only when the first query arrives (:meth:`warmup` moves that cost to
a background thread).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

from src.data_model import SearchState
from src.index import Indexer, JobRegistry, get_status
from src.ingest import VideoIngestor, sanitize_collection_id
from src.query import Search2D
from src.utils.db import META_TABLE, connect

from .cache import SceneCloudCache


class SceneService:
    """Scenes, queries and video ingestion, over one LanceDB store.

    Args:
        config_path: YAML config; the ``ui:`` section configures the cache,
                     ingest directory and cloud resolution.
    """

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config_path = Path(config_path)
        cfg = yaml.safe_load(self.config_path.read_text()) or {}
        ui = cfg.get("ui", {}) or {}

        self._indexing_cfg = cfg.get("indexing") or {}
        self.db = connect(cfg["indexing"]["db_path"])
        self.ingest_dir = Path(ui.get("ingest_dir", "data/scenes"))
        self.cache = SceneCloudCache(
            ui.get("cache_dir", "data/scene_cache"),
            n_frames=ui.get("scene_n_frames", 250),
            voxel=ui.get("voxel", 0.02),
        )
        self.voxel = self.cache.voxel
        # Best SAM confidence below which a match is flagged weak in the UI.
        # SAM already filters at its own threshold, so this sits above that floor.
        self.detection_warn_threshold = ui.get("detection_warn_threshold", 0.6)

        self._pipeline: Optional[Search2D] = None
        self._ingestor: Optional[VideoIngestor] = None
        self._model_lock = threading.Lock()
        # Queries and ingestion both want the GPU; serialise them so a long
        # ingest can't collide with a search mid-forward-pass.
        self._gpu_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Scenes
    # ------------------------------------------------------------------

    def list_scenes(self) -> list[str]:
        """Every indexed collection, minus the calibration side table."""
        return sorted(
            t for t in self.db.list_tables().tables if t != META_TABLE
        )

    def load_scene(
        self, collection_id: str, *, force: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return the scene point cloud ``(points, colors)`` for a collection."""
        return self.cache.get(self.db, collection_id, force=force)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        collection_id: str,
        *,
        mode: Optional[str] = None,
        top_k_final: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
    ) -> SearchState:
        """Run the full search chain and return the finished state.

        ``mode`` / ``top_k_final`` / ``retrieval_mode`` are per-query overrides;
        left ``None`` they fall back to the pipeline's configured defaults from
        ``config.yaml`` (so the app doesn't have to restate them).

        The state carries both the 3D objects (``projected``) and the 2D frames
        that produced them (``results``), so the caller can show the evidence
        behind a highlight.
        """
        pipeline = self._ensure_pipeline()
        with self._gpu_lock:
            return pipeline.invoke(
                query=text,
                collection_id=collection_id,
                mode=mode,
                top_k_final=top_k_final,
                retrieval_mode=retrieval_mode,
            )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_video(self, video_bytes: bytes, filename: str) -> str:
        """Start ingesting an uploaded video; return its job id immediately.

        The work runs on a background thread — extraction, reconstruction and
        indexing take minutes, and the caller (a UI callback) must stay
        responsive. Poll :meth:`job_status` with the returned id.

        Args:
            video_bytes: The uploaded file's contents.
            filename:    Its original name; the collection is named after it.

        Returns:
            The job id.
        """
        collection_id = self._unique_collection_id(sanitize_collection_id(filename))

        scene_dir = self.ingest_dir / collection_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename).suffix or ".mp4"
        video_path = scene_dir / f"video{suffix}"
        video_path.write_bytes(video_bytes)

        job = JobRegistry.start(
            collection_id=collection_id, total=0, stages=VideoIngestor.STAGES
        )
        threading.Thread(
            target=self._run_ingest,
            args=(video_path, collection_id, job),
            daemon=True,
        ).start()
        return job.job_id

    def job_status(self, job_id: str) -> Optional[dict]:
        """Snapshot of a job — state, stage, and progress within that stage."""
        return get_status(job_id)

    def _run_ingest(self, video_path: Path, collection_id: str, job) -> None:
        """Body of the ingest thread; every failure lands on the job.

        Nothing watches this thread, so an escaping exception would leave the
        poller waiting on a job that never moves.
        """
        try:
            ingestor = self._ensure_ingestor()   # may load models; not under _gpu_lock
        except Exception as exc:
            job.fail(f"could not start ingestion: {exc}")
            return

        try:
            with self._gpu_lock:
                ingestor.ingest(video_path, collection_id, job=job)
        except Exception:
            return                               # ingest() already failed the job
        self.cache.invalidate(collection_id)

    def _unique_collection_id(self, base: str) -> str:
        """``base``, or ``base-2``, ``base-3``… so an upload never overwrites a scene."""
        existing = set(self.list_scenes())
        if base not in existing:
            return base
        n = 2
        while f"{base}-{n}" in existing:
            n += 1
        return f"{base}-{n}"

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Load the models now (so the first query isn't the one that waits)."""
        self._ensure_pipeline()

    def _ensure_pipeline(self) -> Search2D:
        with self._model_lock:
            if self._pipeline is None:
                self._pipeline = Search2D.from_config(self.config_path)
            return self._pipeline

    def _ensure_ingestor(self) -> VideoIngestor:
        # Reuse the pipeline's SigLIP rather than loading a second copy onto the
        # GPU — indexing and querying embed with the same model.
        pipeline = self._ensure_pipeline()
        with self._model_lock:
            if self._ingestor is None:
                indexer = Indexer(
                    model=pipeline.embed.siglip,
                    db_path=self._indexing_cfg["db_path"],
                    batch_size=self._indexing_cfg.get("batch_size"),
                )
                self._ingestor = VideoIngestor.from_config(
                    self.config_path, indexer=indexer
                )
            return self._ingestor
