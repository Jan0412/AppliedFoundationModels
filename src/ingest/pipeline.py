"""Video ingestion: an uploaded video becomes a queryable collection.

Three stages, tracked on one :class:`JobStatus` so a UI can show both which
stage is running and how far into it we are::

    extract  →  reconstruct  →  index
    (PyAV)      (VGGT, mocked)   (SigLIP → LanceDB)

The ingestor never touches the UI: it only mutates the job, which any reader
(a viser poller today, an HTTP handler tomorrow) can snapshot with
:func:`~src.index.get_status`.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.index import Indexer, JobRegistry, JobStatus
from src.reconstruct import BaseReconstructor, MockVGGTReconstructor

from .video import extract_frames, probe_frame_count


class VideoIngestor:
    """Drive a video from upload to indexed collection.

    Args:
        indexer:       Writes embeddings to LanceDB. Callers that already hold
                       a loaded SigLIP should pass an :class:`Indexer` built
                       around it rather than loading the model twice.
        reconstructor: Supplies depth + poses for the extracted frames.
        scenes_dir:    Root for per-scene working directories.
        fps:           Frames sampled per second of video.
        max_frames:    Hard cap on frames taken from one video.
    """

    #: Stage names, in order — the job's declared plan.
    STAGES: tuple[str, ...] = ("extract", "reconstruct", "index")

    def __init__(
        self,
        indexer: Indexer,
        reconstructor: BaseReconstructor,
        scenes_dir: str | Path,
        *,
        fps: float = 2.0,
        max_frames: int = 300,
    ) -> None:
        self.indexer = indexer
        self.reconstructor = reconstructor
        self.scenes_dir = Path(scenes_dir)
        self.fps = fps
        self.max_frames = max_frames

    @classmethod
    def from_config(
        cls,
        path: str | Path = "config.yaml",
        *,
        indexer: Indexer | None = None,
    ) -> "VideoIngestor":
        """Build an ingestor from the ``ui:`` section of *path*.

        Args:
            path:    Config file.
            indexer: Optional pre-built indexer (see the constructor); one is
                     created from the config when omitted.
        """
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        ui = cfg.get("ui", {}) or {}

        # TODO: swap for VGGTReconstructor once VGGT is implemented — the
        # FrameSet contract is unchanged, so this line is the whole migration.
        reconstructor = MockVGGTReconstructor()

        return cls(
            indexer=indexer if indexer is not None else Indexer.from_config(path),
            reconstructor=reconstructor,
            scenes_dir=ui.get("ingest_dir", "data/scenes"),
            fps=ui.get("extract_fps", 2.0),
            max_frames=ui.get("extract_max_frames", 300),
        )

    def scene_dir(self, collection_id: str) -> Path:
        """Working directory holding the video and its extracted frames."""
        return self.scenes_dir / collection_id

    def ingest(
        self,
        video_path: str | Path,
        collection_id: str,
        *,
        job: JobStatus | None = None,
    ) -> JobStatus:
        """Extract, reconstruct and index *video_path* as *collection_id*.

        Runs synchronously — callers wanting a responsive UI run this in a
        thread and poll *job*.

        Args:
            video_path:    The uploaded video.
            collection_id: Name for the new collection. Must not already exist
                           (the indexer rejects duplicate ids).
            job:           Optional pre-registered job, so a caller can hold the
                           id before the work starts. Created here otherwise.

        Returns:
            The job, in state ``"done"`` — or ``"failed"`` if it raised.
        """
        job = job or JobRegistry.start(
            collection_id=collection_id, total=0, stages=self.STAGES
        )
        frames_dir = self.scene_dir(collection_id) / "rgb"

        try:
            planned = probe_frame_count(video_path, fps=self.fps) or self.max_frames
            job.set_stage("extract", total=min(planned, self.max_frames))
            rgb_paths = extract_frames(
                video_path,
                frames_dir,
                fps=self.fps,
                max_frames=self.max_frames,
                on_progress=job.advance,
            )
            if not rgb_paths:
                raise ValueError(f"no frames could be decoded from {video_path}")

            job.set_stage("reconstruct", total=len(rgb_paths))
            frames = self.reconstructor.reconstruct(
                rgb_paths, self.scene_dir(collection_id), on_progress=job.advance
            )

            job.set_stage("index", total=len(frames))
            self.indexer.insert(
                frames.paths,
                collection_id,
                depth_paths=frames.depth_paths,
                poses=frames.poses,
                intrinsics=frames.intrinsics,
                depth_scale=frames.depth_scale,
                job=job,
            )
            job.finish()
        except Exception as exc:
            job.fail(str(exc))
            raise

        return job


def sanitize_collection_id(name: str) -> str:
    """Turn a filename into a collection id LanceDB and the cache can hold.

    Table names and cache filenames both come from this, so anything outside
    ``[A-Za-z0-9_-]`` is collapsed to an underscore.
    """
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return cleaned or "scene"
