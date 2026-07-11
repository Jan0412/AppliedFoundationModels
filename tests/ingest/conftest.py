"""Shared fixtures for tests/ingest/.

tiny_video   – factory encoding a real mp4 with PyAV (no fixture files in git,
               and it exercises the same decode path production uses).
scannet_tree – synthetic ScanNet sequence for the mock reconstructor.
mock_indexer – Indexer on a tmp LanceDB with a mocked SigLIP (real DB writes,
               no model download).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import av
import numpy as np
import pytest
from PIL import Image

from src.index import Indexer

EMBED_DIM = 8


@pytest.fixture
def tiny_video(tmp_path):
    def _make(n_frames: int = 16, fps: int = 8, size: int = 64, name: str = "clip.mp4"):
        path = tmp_path / name
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=fps)
            stream.width = stream.height = size
            stream.pix_fmt = "yuv420p"

            for i in range(n_frames):
                # A moving bright block, so frames differ from one another.
                img = np.zeros((size, size, 3), dtype=np.uint8)
                x = (i * 4) % (size - 8)
                img[x : x + 8, x : x + 8] = 255
                frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                container.mux(stream.encode(frame))

            container.mux(stream.encode(None))   # flush
        return path

    return _make


@pytest.fixture
def scannet_tree(tmp_path):
    def _make(n: int = 5):
        root = tmp_path / "scene"
        for sub in ("color", "depth", "pose", "intrinsic"):
            (root / sub).mkdir(parents=True, exist_ok=True)

        for i in range(n):
            Image.new("RGB", (8, 8), color=(i, 0, 0)).save(root / "color" / f"{i}.jpg")
            Image.fromarray(np.full((8, 8), 1000, dtype=np.uint16)).save(
                root / "depth" / f"{i}.png"
            )
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = float(i)
            np.savetxt(root / "pose" / f"{i}.txt", pose)

        K = np.eye(4)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = 100.0, 100.0, 4.0, 4.0
        np.savetxt(root / "intrinsic" / "intrinsic_depth.txt", K)
        return root

    return _make


@pytest.fixture
def mock_siglip_model():
    model = MagicMock()
    model.batch_size = 4
    model.embedding_dim = EMBED_DIM
    model.embed_images.side_effect = lambda imgs: np.tile(
        np.eye(1, EMBED_DIM, dtype=np.float32), (len(imgs), 1)
    )
    return model


@pytest.fixture
def mock_indexer(tmp_path, mock_siglip_model):
    return Indexer(model=mock_siglip_model, db_path=tmp_path / "lancedb")
