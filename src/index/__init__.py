"""Image indexing pipeline: SigLIP embeddings → LanceDB.

Usage::

    from src.index import Indexer, get_status

    idx = Indexer.from_config("config.yaml")

    # First-time index (raises ValueError if any id already exists)
    job = idx.insert(image_paths, collection_id="fr1_desk")

    # Re-index / update existing rows (upsert semantics)
    job2 = idx.update(image_paths, collection_id="fr1_desk")

    # Query status (e.g. from an API handler)
    print(get_status(job.job_id))
"""

from .indexer import Indexer
from .status import JobRegistry, JobStatus, get_status

__all__ = ["Indexer", "JobStatus", "JobRegistry", "get_status"]
