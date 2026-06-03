"""Image indexing pipeline: SigLIP embeddings → LanceDB.

Usage::

    from src.index import Indexer, ObjectExtractor, get_status

    pre = ObjectExtractor.from_config("config.yaml")
    idx = Indexer.from_config("config.yaml")

    # Preprocess: remove background and centre each object before embedding
    clean = [pre.invoke(img) for img in raw_pil_images]

    # First-time index (raises ValueError if any id already exists)
    job = idx.insert(clean, collection_id="fr1_desk", ids=[...])

    # Re-index / update existing rows (upsert semantics)
    job2 = idx.update(clean, collection_id="fr1_desk")

    # Query status (e.g. from an API handler)
    print(get_status(job.job_id))
"""

from .indexer import Indexer
from .preprocess import ObjectExtractor
from .status import JobRegistry, JobStatus, get_status

__all__ = ["Indexer", "ObjectExtractor", "JobStatus", "JobRegistry", "get_status"]
