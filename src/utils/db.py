"""LanceDB connection helper.

Single entry point so both the indexer and the query pipeline open the
store the same way (and tests can monkey-patch one function).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lancedb

#: Side table written by :class:`src.index.Indexer` holding one calibration
#: row (intrinsics + depth_scale) per collection. Kept here too so the query
#: side can read it without importing the indexer.
META_TABLE = "_collection_meta"


def connect(db_path: str | Path) -> lancedb.DBConnection:
    """Open (or create) a LanceDB store at *db_path*.

    Args:
        db_path: Directory that LanceDB uses as its root store.

    Returns:
        An open :class:`lancedb.DBConnection`.
    """
    return lancedb.connect(str(db_path))


def load_collection_meta(db, collection_id: str) -> Optional[dict]:
    """Return the calibration row for *collection_id*, or ``None`` if absent.

    Reads the single row keyed by ``collection_id`` from :data:`META_TABLE`.

    Args:
        db:            An open :class:`lancedb.DBConnection`.
        collection_id: The collection whose calibration to fetch.

    Returns:
        A dict ``{"fx", "fy", "cx", "cy", "depth_scale"}`` (floats), or
        ``None`` when no metadata table or matching row exists.
    """
    if META_TABLE not in db.list_tables().tables:
        return None
    rows = (
        db.open_table(META_TABLE)
        .search()
        .where(f"collection_id = '{collection_id}'")
        .limit(1)
        .to_list()
    )
    if not rows:
        return None
    r = rows[0]
    return {k: float(r[k]) for k in ("fx", "fy", "cx", "cy", "depth_scale")}
