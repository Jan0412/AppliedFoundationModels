"""Shared fixtures for tests/ui/.

populated_store – a tmp LanceDB with one small, fully-projectable collection
                  plus its calibration row (the shape Indexer writes), so cache
                  and service tests run against a real store with no models.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from src.utils.db import connect

EMBED_DIM = 8


@pytest.fixture
def populated_store(tmp_path):
    """Return (db, collection_id) for a 3-frame projectable collection."""
    depth_dir = tmp_path / "depth"
    rgb_dir = tmp_path / "rgb"
    depth_dir.mkdir()
    rgb_dir.mkdir()

    paths, depth_paths = [], []
    for i in range(3):
        rp = rgb_dir / f"{i}.png"
        Image.new("RGB", (8, 8), color=(i * 40, 0, 0)).save(rp)
        paths.append(str(rp))
        dp = depth_dir / f"{i}.png"
        Image.fromarray(np.full((8, 8), 1000, dtype=np.uint16)).save(dp)
        depth_paths.append(str(dp))

    identity16 = np.eye(4, dtype=np.float32).reshape(16).tolist()
    db = connect(tmp_path / "lancedb")
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("collection_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
        pa.field("path", pa.string()),
        pa.field("depth_path", pa.string()),
        pa.field("cam2world", pa.list_(pa.float32())),
    ])
    table = db.create_table("room", schema=schema)
    table.add([
        {
            "id": f"id-{i}",
            "collection_id": "room",
            "vector": np.eye(1, EMBED_DIM, dtype=np.float32)[0].tolist(),
            "path": paths[i],
            "depth_path": depth_paths[i],
            "cam2world": identity16,
        }
        for i in range(3)
    ])

    meta_schema = pa.schema([
        pa.field("collection_id", pa.string()),
        pa.field("fx", pa.float64()), pa.field("fy", pa.float64()),
        pa.field("cx", pa.float64()), pa.field("cy", pa.float64()),
        pa.field("depth_scale", pa.float64()),
    ])
    meta = db.create_table("_collection_meta", schema=meta_schema)
    meta.add([{
        "collection_id": "room",
        "fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 4.0, "depth_scale": 1000.0,
    }])

    return db, "room"
