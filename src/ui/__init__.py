"""Web UI: a viser 3D scene viewer with text query and video ingestion.

:class:`SceneService` is the framework-agnostic core (scenes, queries, ingest);
:mod:`src.ui.app` is the viser shell over it.
"""

from .cache import SceneCloudCache
from .service import SceneService

__all__ = ["SceneService", "SceneCloudCache"]
