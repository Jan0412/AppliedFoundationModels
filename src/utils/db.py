"""LanceDB connection helper.

Single entry point so both the indexer and the query pipeline open the
store the same way (and tests can monkey-patch one function).
"""

from __future__ import annotations

from pathlib import Path

import lancedb


def connect(db_path: str | Path) -> lancedb.DBConnection:
    """Open (or create) a LanceDB store at *db_path*.

    Args:
        db_path: Directory that LanceDB uses as its root store.

    Returns:
        An open :class:`lancedb.DBConnection`.
    """
    return lancedb.connect(str(db_path))
