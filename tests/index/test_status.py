"""Tests for src.index.status."""

from __future__ import annotations

import threading
import time

import pytest

from src.index.status import JobRegistry, JobStatus, get_status


# ---------------------------------------------------------------------------
# JobStatus
# ---------------------------------------------------------------------------


def test_jobstatus_initial_state():
    """Newly created JobStatus is 'running' with processed=0 and started_at set."""
    s = JobStatus(job_id="j1", collection_id="c1", total=10)
    assert s.job_id == "j1"
    assert s.collection_id == "c1"
    assert s.total == 10
    assert s.processed == 0
    assert s.state == "running"
    assert isinstance(s.started_at, float)
    assert s.started_at > 0
    assert s.finished_at is None
    assert s.error is None


def test_jobstatus_advance_increments_processed():
    s = JobStatus(job_id="j", collection_id="c", total=100)
    s.advance(7)
    assert s.processed == 7
    s.advance(3)
    assert s.processed == 10


def test_jobstatus_finish_sets_state_done_and_finished_at():
    s = JobStatus(job_id="j", collection_id="c", total=5)
    started = s.started_at
    time.sleep(0.001)
    s.finish()
    assert s.state == "done"
    assert isinstance(s.finished_at, float)
    assert s.finished_at >= started
    assert s.error is None


def test_jobstatus_fail_sets_state_failed_with_error():
    s = JobStatus(job_id="j", collection_id="c", total=5)
    s.fail("boom")
    assert s.state == "failed"
    assert s.error == "boom"
    assert isinstance(s.finished_at, float)


def test_jobstatus_as_dict_returns_all_public_fields():
    """as_dict returns exactly the eight documented keys and no private ones."""
    s = JobStatus(job_id="abc", collection_id="cid", total=4)
    s.advance(2)
    d = s.as_dict()
    expected_keys = {
        "job_id", "collection_id", "state", "total",
        "processed", "started_at", "finished_at", "error",
    }
    assert set(d.keys()) == expected_keys
    assert d["job_id"] == "abc"
    assert d["processed"] == 2
    assert d["state"] == "running"
    # No private keys leak
    assert not any(k.startswith("_") for k in d)


def test_jobstatus_as_dict_after_finish():
    s = JobStatus(job_id="j", collection_id="c", total=2)
    s.advance(2)
    s.finish()
    d = s.as_dict()
    assert d["state"] == "done"
    assert d["processed"] == 2
    assert d["finished_at"] is not None
    assert d["error"] is None


def test_jobstatus_repr_returns_string():
    """__repr__ must return a *str* — not None (kills 'return → None' mutant)."""
    s = JobStatus(job_id="J", collection_id="C", total=10)
    # Call __repr__ directly so we can assert on the return value itself;
    # repr() raises TypeError instead of failing an assertion when the
    # method returns None, which some mutation tools score as "survived".
    result = s.__repr__()
    assert isinstance(result, str), "__repr__ must return str, not None"
    assert result, "__repr__ must return a non-empty string"


def test_jobstatus_repr_includes_progress():
    s = JobStatus(job_id="J", collection_id="C", total=10)
    s.advance(3)
    # Call __repr__ directly so an assertion (not a TypeError) fails when
    # the method returns None — that kills the 'return value → None' mutant.
    r = s.__repr__()
    assert isinstance(r, str), "__repr__ must return str"
    assert "J" in r and "C" in r and "3/10" in r and "running" in r


def test_jobstatus_advance_is_thread_safe():
    """Concurrent advance() calls do not lose updates."""
    s = JobStatus(job_id="j", collection_id="c", total=10_000)
    n_threads, per_thread = 10, 1_000

    def _worker():
        for _ in range(per_thread):
            s.advance(1)

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert s.processed == n_threads * per_thread


# ---------------------------------------------------------------------------
# JobRegistry
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear the registry around each test to avoid bleed-through."""
    JobRegistry._jobs.clear()
    yield
    JobRegistry._jobs.clear()


def test_registry_start_creates_and_registers_job():
    job = JobRegistry.start(collection_id="c1", total=5)
    assert job.collection_id == "c1"
    assert job.total == 5
    assert job.state == "running"
    assert JobRegistry.get(job.job_id) is job


def test_registry_start_uses_supplied_job_id():
    job = JobRegistry.start(collection_id="c", total=1, job_id="my-id")
    assert job.job_id == "my-id"
    assert JobRegistry.get("my-id") is job


def test_registry_start_generates_job_id_when_none():
    job = JobRegistry.start(collection_id="c", total=1)
    assert isinstance(job.job_id, str)
    assert len(job.job_id) > 0
    # Two consecutive jobs get different ids
    other = JobRegistry.start(collection_id="c", total=1)
    assert other.job_id != job.job_id


def test_registry_get_returns_none_for_unknown_id():
    assert JobRegistry.get("does-not-exist") is None


def test_registry_list_all_returns_snapshot_of_all_jobs():
    a = JobRegistry.start(collection_id="a", total=1)
    b = JobRegistry.start(collection_id="b", total=1)
    listed = JobRegistry.list_all()
    assert {j.job_id for j in listed} == {a.job_id, b.job_id}


def test_registry_list_all_is_a_snapshot_not_a_live_view():
    JobRegistry.start(collection_id="a", total=1)
    snapshot = JobRegistry.list_all()
    JobRegistry.start(collection_id="b", total=1)
    # The pre-existing snapshot does not grow
    assert len(snapshot) == 1


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


def test_get_status_returns_dict_for_known_job():
    job = JobRegistry.start(collection_id="c", total=3, job_id="x")
    job.advance(1)
    d = get_status("x")
    assert isinstance(d, dict)
    assert d["job_id"] == "x"
    assert d["processed"] == 1


def test_get_status_returns_none_for_unknown_job():
    assert get_status("nope") is None
