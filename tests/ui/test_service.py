"""Tests for src.ui.service.SceneService (no models, no viser).

Only the framework-agnostic glue is exercised: scene listing, collection-id
uniquification, job pass-through. Query and ingest drive real models and are
covered where those live (tests/query, tests/ingest).
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.ui.service import SceneService


@pytest.fixture
def service(tmp_path, populated_store, monkeypatch):
    """A SceneService pointed at the populated tmp store via a tmp config."""
    db, _ = populated_store
    db_path = db.uri

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "indexing": {"db_path": db_path},
        "ui": {
            "cache_dir": str(tmp_path / "cache"),
            "ingest_dir": str(tmp_path / "scenes"),
            "voxel": 0.02,
            "scene_n_frames": 10,
        },
    }))
    return SceneService(config)


def test_list_scenes_filters_meta_table(service):
    scenes = service.list_scenes()
    assert scenes == ["room"]
    assert "_collection_meta" not in scenes


def test_load_scene_returns_a_cloud(service):
    points, _ = service.load_scene("room")
    assert points.shape[1] == 3
    assert len(points) > 0


def test_job_status_passes_through(service):
    from src.index import JobRegistry
    job = JobRegistry.start("room", total=0, stages=("index",))
    assert service.job_status(job.job_id)["job_id"] == job.job_id
    assert service.job_status("nope") is None


def test_unique_collection_id_avoids_existing_scenes(service):
    # "room" already exists → suffixes; a fresh name is returned unchanged.
    assert service._unique_collection_id("room") == "room-2"
    assert service._unique_collection_id("kitchen") == "kitchen"


def test_ingest_video_is_non_blocking_and_writes_the_upload(service, tmp_path, monkeypatch):
    """ingest_video returns a job id at once and persists the bytes; the heavy
    work is stubbed so no models load."""
    started = {}

    def _fake_run(video_path, collection_id, job):
        started["path"] = video_path
        started["cid"] = collection_id
        job.finish()

    monkeypatch.setattr(service, "_run_ingest", _fake_run)

    job_id = service.ingest_video(b"fake-bytes", "My Clip.mp4")

    assert isinstance(job_id, str)
    # Give the daemon thread a moment to run the stub.
    import time
    for _ in range(50):
        if "cid" in started:
            break
        time.sleep(0.01)
    assert started["cid"] == "My_Clip"
    assert started["path"].read_bytes() == b"fake-bytes"
    assert started["path"].suffix == ".mp4"
