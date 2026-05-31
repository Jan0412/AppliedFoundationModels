"""Job status tracking for the image indexer.

Thread-safe by design so the synchronous ``Indexer`` can later be moved to a
background thread (e.g. when wired up to an HTTP API) without changing the
reader side.

Typical use::

    from src.index.status import get_status

    # Inside the API handler:
    status = get_status(job_id)   # returns a plain dict, or None if unknown
"""

from __future__ import annotations

import threading
import time
import uuid


class JobStatus:
    """Mutable, thread-safe status record for a single indexing job."""

    def __init__(
        self,
        job_id: str,
        collection_id: str,
        total: int,
    ) -> None:
        self.job_id = job_id
        self.collection_id = collection_id
        self.total = total
        self.processed: int = 0
        self.state: str = "running"       # "running" | "done" | "failed"
        self.started_at: float = time.time()
        self.finished_at: float | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutators (safe to call from the indexing thread)
    # ------------------------------------------------------------------

    def advance(self, n: int) -> None:
        """Increment the processed-frame counter by *n*."""
        with self._lock:
            self.processed += n

    def finish(self) -> None:
        """Mark the job as successfully completed."""
        with self._lock:
            self.state = "done"
            self.finished_at = time.time()

    def fail(self, msg: str) -> None:
        """Mark the job as failed with an error message."""
        with self._lock:
            self.state = "failed"
            self.error = msg
            self.finished_at = time.time()

    # ------------------------------------------------------------------
    # Serialisation (safe to call from any thread / API handler)
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of this status."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "collection_id": self.collection_id,
                "state": self.state,
                "total": self.total,
                "processed": self.processed,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
            }

    def __repr__(self) -> str:
        return (
            f"JobStatus(job_id={self.job_id!r}, collection_id={self.collection_id!r}, "
            f"state={self.state!r}, processed={self.processed}/{self.total})"
        )


class JobRegistry:
    """Module-level registry of all indexing jobs, keyed by job_id.

    Thread-safe: all mutations and reads hold ``_lock``.
    """

    _jobs: dict[str, JobStatus] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def start(
        cls,
        collection_id: str,
        total: int,
        job_id: str | None = None,
    ) -> JobStatus:
        """Create a new :class:`JobStatus`, register it, and return it.

        Args:
            collection_id: The LanceDB table / collection being indexed.
            total:         Total number of images in this job.
            job_id:        Optional caller-supplied id (e.g. a request UUID).
                           If omitted, a random hex id is generated.
        """
        job_id = job_id or uuid.uuid4().hex
        job = JobStatus(job_id=job_id, collection_id=collection_id, total=total)
        with cls._lock:
            cls._jobs[job_id] = job
        return job

    @classmethod
    def get(cls, job_id: str) -> JobStatus | None:
        """Return the :class:`JobStatus` for *job_id*, or ``None``."""
        with cls._lock:
            return cls._jobs.get(job_id)

    @classmethod
    def list_all(cls) -> list[JobStatus]:
        """Return a snapshot list of all known jobs."""
        with cls._lock:
            return list(cls._jobs.values())


# ---------------------------------------------------------------------------
# Convenience function — the surface the API layer will call
# ---------------------------------------------------------------------------

def get_status(job_id: str) -> dict | None:
    """Return the status of *job_id* as a plain dict, or ``None`` if unknown.

    This is the only function the HTTP API needs to import::

        from src.index import get_status
        return get_status(job_id) or abort(404)
    """
    job = JobRegistry.get(job_id)
    return job.as_dict() if job is not None else None
