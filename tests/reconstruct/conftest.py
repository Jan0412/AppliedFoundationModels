"""Shared fixtures for tests/reconstruct/.

scannet_tree – factory writing a synthetic ScanNet sequence (color/depth/pose/
               intrinsic) into a tmp dir, so the mock reconstructor is never
               pointed at the real dataset mirror.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def scannet_tree(tmp_path):
    def _make(n: int = 5, root_name: str = "scene"):
        root = tmp_path / root_name
        for sub in ("color", "depth", "pose", "intrinsic"):
            (root / sub).mkdir(parents=True, exist_ok=True)

        for i in range(n):
            Image.new("RGB", (8, 8), color=(i, 0, 0)).save(root / "color" / f"{i}.jpg")
            Image.fromarray(np.full((8, 8), 1000, dtype=np.uint16)).save(
                root / "depth" / f"{i}.png"
            )
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = float(i)          # cameras spread along +x
            np.savetxt(root / "pose" / f"{i}.txt", pose)

        K = np.eye(4)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = 100.0, 100.0, 4.0, 4.0
        np.savetxt(root / "intrinsic" / "intrinsic_depth.txt", K)
        return root

    return _make
