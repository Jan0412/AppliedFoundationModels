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


def test_unique_collection_id_keeps_counting_past_the_first_suffix(service, monkeypatch):
    """Upload the same clip three times and the third must not clobber the second."""
    monkeypatch.setattr(
        service, "list_scenes", lambda: ["room", "room-2", "room-3"]
    )
    assert service._unique_collection_id("room") == "room-4"


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


# ---------------------------------------------------------------------------
# Live settings
# ---------------------------------------------------------------------------


class _Step:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakePipeline:
    """Search2D's shape with no models, so query() can run in a test."""

    def __init__(self):
        self.retrieve = _Step(mode="dynamic", min_k=50, max_k=200,
                              min_separability=0.75, strategy="tail")
        self.select = _Step(n_diverse=10, patch_frac=0.2)
        self.rerank = _Step(top_k_final=10)
        self.project = _Step(mode="cluster_single", max_instances=None,
                             min_instance_size=50, min_views=1, voxel=0.01,
                             cluster_eps=0.05, cluster_min_samples=10,
                             bbox_percentile=(2.0, 98.0))
        self.invoked_with: dict = {}

    def invoke(self, **kw):
        self.invoked_with = kw
        return "state"


def test_settings_start_from_the_config_defaults(service):
    values = service.settings()
    assert values["mode"] == "cluster_single"
    assert values["min_views"] == 1


def test_update_settings_stages_without_touching_the_pipeline(service):
    service.update_settings({"min_views": 4})
    assert service.settings()["min_views"] == 4
    # Staged only — no pipeline was built, so no models were loaded.
    assert service._pipeline is None


def test_query_applies_staged_settings_to_the_pipeline(service):
    pipeline = _FakePipeline()
    service._pipeline = pipeline          # skip model loading
    service.update_settings({
        "min_views": 4, "mode": "cluster_instances",
        "retrieval_mode": "topk", "n_diverse": 6,
    })

    service.query("chair", "room")

    assert pipeline.project.min_views == 4
    assert pipeline.project.mode == "cluster_instances"
    assert pipeline.retrieve.mode == "topk"        # not project.mode
    assert pipeline.select.n_diverse == 6


def test_reset_settings_rereads_the_config(service):
    service.update_settings({"min_views": 9})
    values = service.reset_settings()
    assert values["min_views"] == 1
    assert service.settings()["min_views"] == 1


# ---------------------------------------------------------------------------
# The ingest thread
#
# Nothing joins this thread, so every failure has to land on the job — a poller
# waiting on a job that never moves is the bug these pin down.
# ---------------------------------------------------------------------------


class _FakeJob:
    def __init__(self):
        self.error = None
        self.finished = False

    def fail(self, message):
        self.error = message

    def finish(self):
        self.finished = True


class _FakeIngestor:
    def __init__(self, exc: Exception | None = None):
        self.exc = exc
        self.calls: list[tuple] = []

    def ingest(self, video_path, collection_id, job=None):
        self.calls.append((video_path, collection_id))
        if self.exc is not None:
            job.fail(str(self.exc))      # the real ingestor fails the job, then raises
            raise self.exc


def test_run_ingest_invalidates_the_cache_on_success(service, tmp_path, monkeypatch):
    """A re-ingested collection must not keep serving the old cloud from cache."""
    ingestor = _FakeIngestor()
    monkeypatch.setattr(service, "_ensure_ingestor", lambda: ingestor)
    invalidated: list[str] = []
    monkeypatch.setattr(service.cache, "invalidate", invalidated.append)

    job = _FakeJob()
    service._run_ingest(tmp_path / "video.mp4", "clip", job)

    assert ingestor.calls == [(tmp_path / "video.mp4", "clip")]
    assert invalidated == ["clip"]
    assert job.error is None


def test_a_model_load_failure_lands_on_the_job(service, tmp_path, monkeypatch):
    def _boom():
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(service, "_ensure_ingestor", _boom)

    job = _FakeJob()
    service._run_ingest(tmp_path / "video.mp4", "clip", job)   # must not raise

    assert "could not start ingestion" in job.error
    assert "no CUDA device" in job.error


def test_a_failed_ingest_leaves_the_cache_alone(service, tmp_path, monkeypatch):
    """ingest() already failed the job; the half-written collection stays uncached."""
    ingestor = _FakeIngestor(exc=ValueError("no frames could be decoded"))
    monkeypatch.setattr(service, "_ensure_ingestor", lambda: ingestor)
    invalidated: list[str] = []
    monkeypatch.setattr(service.cache, "invalidate", invalidated.append)

    job = _FakeJob()
    service._run_ingest(tmp_path / "video.mp4", "clip", job)   # must not raise

    assert job.error == "no frames could be decoded"
    assert invalidated == []


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------


def test_the_pipeline_is_built_once_and_reused(service, monkeypatch):
    """Construction must not load models — the first query (or warmup) does."""
    import src.ui.service as service_mod

    builds = []
    pipeline = _FakePipeline()
    monkeypatch.setattr(
        service_mod.Search2D, "from_config",
        lambda path: (builds.append(path), pipeline)[1],
    )

    assert service._pipeline is None      # nothing loaded at construction
    service.warmup()
    service.warmup()

    assert service._pipeline is pipeline
    assert builds == [service.config_path]     # loaded exactly once


def test_the_ingestor_reuses_the_pipelines_embedder(service, monkeypatch):
    """Indexing and querying embed with the same model — loading a second copy
    onto the GPU is what this sharing avoids."""
    import src.ui.service as service_mod

    pipeline = _FakePipeline()
    pipeline.embed = _Step(embedder="the-one-embedder")
    service._pipeline = pipeline

    seen = {}

    def _fake_indexer(model, db_path, batch_size):
        seen["model"] = model
        seen["db_path"] = db_path
        return "indexer"

    def _fake_from_config(path, indexer):
        seen["indexer"] = indexer
        return "ingestor"

    monkeypatch.setattr(service_mod, "Indexer", _fake_indexer)
    monkeypatch.setattr(service_mod.VideoIngestor, "from_config", _fake_from_config)

    assert service._ensure_ingestor() == "ingestor"
    assert service._ensure_ingestor() == "ingestor"     # cached, not rebuilt

    assert seen["model"] == "the-one-embedder"
    assert seen["indexer"] == "indexer"
