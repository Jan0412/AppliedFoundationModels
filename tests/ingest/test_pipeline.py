"""Tests for src.ingest.pipeline (VideoIngestor end-to-end, with mocks).

A real mp4 and a real LanceDB, a mocked SigLIP and a mock reconstructor: the
orchestration — stage bookkeeping, disk layout, what lands in the DB — is what
these pin down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.index import JobRegistry, get_status
from src.ingest import VideoIngestor, sanitize_collection_id
from src.reconstruct import MockVGGTReconstructor
from src.utils.datasets import load_collection
from src.utils.db import load_collection_meta


@pytest.fixture
def ingestor(mock_indexer, scannet_tree, tmp_path):
    return VideoIngestor(
        indexer=mock_indexer,
        reconstructor=MockVGGTReconstructor(scene_dir=scannet_tree(n=6)),
        scenes_dir=tmp_path / "scenes",
        fps=None,
        max_frames=50,
    )


def test_ingest_produces_a_queryable_collection(ingestor, mock_indexer, tiny_video):
    video = tiny_video(n_frames=6, fps=6)

    job = ingestor.ingest(video, "clip")

    assert job.state == "done"

    # The collection reads back as a FrameSet — the same shape a benchmark loader
    # would have produced, so the query pipeline can project it to 3D.
    frames = load_collection(mock_indexer.db, "clip")
    assert len(frames) == 6
    assert all(Path(p).is_file() for p in frames.depth_paths)
    assert load_collection_meta(mock_indexer.db, "clip") is not None


def test_ingest_walks_all_three_stages(ingestor, tiny_video):
    """The job reports which stage is running, so the UI can show 'step 3/3'."""
    job = ingestor.ingest(tiny_video(n_frames=4, fps=4), "clip")

    assert job.stages == VideoIngestor.STAGES
    assert job.stage == "index"          # the last stage entered
    assert job.processed == job.total    # and it ran to completion


def test_each_stage_starts_with_its_own_reset_counter(ingestor, tiny_video, monkeypatch):
    """A reader polling get_status sees the stage change and the counter restart.

    Without the reset, progress would keep climbing across stages and a
    percentage bar would read 300% by the end.
    """
    job = JobRegistry.start("clip", total=0, stages=VideoIngestor.STAGES)
    snapshots: list[tuple[str, int, int]] = []

    def _snapshot_then(fn):
        def _wrapped(*args, **kwargs):
            d = get_status(job.job_id)
            snapshots.append((d["stage"], d["processed"], d["total"]))
            return fn(*args, **kwargs)
        return _wrapped

    monkeypatch.setattr(
        ingestor.reconstructor, "reconstruct",
        _snapshot_then(ingestor.reconstructor.reconstruct),
    )
    monkeypatch.setattr(
        ingestor.indexer, "insert", _snapshot_then(ingestor.indexer.insert)
    )

    ingestor.ingest(tiny_video(n_frames=4, fps=4), "clip", job=job)

    # Each snapshot is taken as its stage begins: named, zeroed, and sized to the
    # 4 frames extract produced.
    assert snapshots == [("reconstruct", 0, 4), ("index", 0, 4)]


def test_ingest_extracts_frames_into_the_scene_dir(ingestor, tiny_video):
    ingestor.ingest(tiny_video(n_frames=3, fps=3), "clip")
    frames = sorted((ingestor.scene_dir("clip") / "rgb").glob("*.jpg"))
    assert len(frames) == 3


def test_failed_ingest_marks_the_job_failed_and_reraises(ingestor, tiny_video, tmp_path):
    """A missing mock scene surfaces as a failed job, not a silent empty scene."""
    ingestor.reconstructor = MockVGGTReconstructor(scene_dir=tmp_path / "gone")
    job = JobRegistry.start("clip", total=0, stages=VideoIngestor.STAGES)

    with pytest.raises(FileNotFoundError):
        ingestor.ingest(tiny_video(n_frames=2, fps=2), "clip", job=job)

    status = get_status(job.job_id)
    assert status["state"] == "failed"
    assert "VGGT" in status["error"]


def test_reingesting_the_same_collection_id_is_rejected(ingestor, tiny_video):
    """Duplicate ids would silently corrupt a scene — the indexer refuses them."""
    video = tiny_video(n_frames=3, fps=3)
    ingestor.ingest(video, "clip")

    with pytest.raises(ValueError, match="already exist"):
        ingestor.ingest(video, "clip")


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("My Kitchen.mp4", "My_Kitchen"),
        ("/tmp/living room (2).mov", "living_room_2"),
        ("...", "scene"),
        ("scene0011.mp4", "scene0011"),
    ],
)
def test_sanitize_collection_id(filename, expected):
    assert sanitize_collection_id(filename) == expected
