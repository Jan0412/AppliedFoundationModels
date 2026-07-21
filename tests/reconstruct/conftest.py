"""Shared fixtures for tests/reconstruct/.

scannet_tree – factory writing a synthetic ScanNet sequence (color/depth/pose/
               intrinsic) into a tmp dir, so the mock reconstructor is never
               pointed at the real dataset mirror.

rgb_frames – factory writing tiny RGB JPEGs for VGGT-Omega tests.

mock_vggt_omega_patches – stubs HF download / VGGT-Omega / preprocess / pose
                          decode so tests never touch the real checkpoint.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

#: Spatial size of the fake VGGT depth / confidence maps.
FAKE_HW = (4, 4)


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


@pytest.fixture
def rgb_frames(tmp_path):
    """Factory writing *n* tiny RGB JPEGs; returns their paths in order."""

    def _make(n: int = 3):
        root = tmp_path / "rgb"
        root.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i in range(n):
            path = root / f"{i:06d}.jpg"
            Image.new("RGB", (8, 8), color=(i, 0, 0)).save(path)
            paths.append(str(path))
        return paths

    return _make


def _fake_predictions(images: torch.Tensor) -> dict:
    """Predictions shaped like VGGT-Omega for an ``(N, 3, H, W)`` batch."""
    n = int(images.shape[0])
    h, w = FAKE_HW
    depth = torch.full((1, n, h, w, 1), 2.0)          # metres (up-to-scale)
    depth_conf = torch.full((1, n, h, w), 100.0)      # above default threshold
    # Mark one low-confidence pixel so _save_depth_png filtering is exercised.
    depth_conf[:, :, 0, 0] = 1.0
    return {
        "depth": depth,
        "depth_conf": depth_conf,
        "pose_enc": torch.zeros(1, n, 9),
        "images": images.unsqueeze(0),                # (1, N, 3, H, W) → [-2:]=H,W
    }


def _fake_encoding_to_camera(pose_enc: torch.Tensor, hw):  # noqa: ARG001
    n = int(pose_enc.shape[1])
    extrinsics = torch.zeros(1, n, 3, 4)
    intrinsics = torch.zeros(1, n, 3, 3)
    for i in range(n):
        extrinsics[0, i, :3, :3] = torch.eye(3)
        extrinsics[0, i, 0, 3] = float(i)             # camera-from-world +x
        intrinsics[0, i] = torch.tensor(
            [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
        )
    return extrinsics, intrinsics


@pytest.fixture
def mock_vggt_omega_patches(monkeypatch, tmp_path):
    """Patch VGGT-Omega load + inference in ``src.reconstruct.vggt_omega``.

    Returns a dict with the mock model and callables for inspection.
    """
    ckpt = tmp_path / "vggt_omega_fake.pt"
    ckpt.write_bytes(b"x")

    model = MagicMock(name="VGGTOmega")
    model.to.return_value = model
    model.eval.return_value = model
    model.side_effect = _fake_predictions

    model_cls = MagicMock(name="VGGTOmegaCls", return_value=model)

    def _preprocess(paths, mode, image_resolution):  # noqa: ARG001
        n = len(paths)
        h, w = FAKE_HW
        return torch.zeros(n, 3, h, w)

    monkeypatch.setattr("src.reconstruct.vggt_omega.hf_hub_download", lambda **kw: str(ckpt))
    monkeypatch.setattr("src.reconstruct.vggt_omega.torch.load", lambda *a, **k: {})
    monkeypatch.setattr("src.reconstruct.vggt_omega.VGGTOmega", model_cls)
    monkeypatch.setattr(
        "src.reconstruct.vggt_omega.load_and_preprocess_images", _preprocess
    )
    monkeypatch.setattr(
        "src.reconstruct.vggt_omega.encoding_to_camera", _fake_encoding_to_camera
    )

    return {"model": model, "VGGTOmega": model_cls, "checkpoint": ckpt}


#: Constructor kwargs matching ``models.vggt_omega`` (tests use CPU).
VGGT_OMEGA_KWARGS = {
    "model_id": "facebook/VGGT-Omega",
    "checkpoint": "vggt_omega_fake.pt",
    "device": "cpu",
    "image_resolution": 512,
    "preprocess_mode": "balanced",
    "confidence_threshold": 10.0,
    "depth_scale": 1000.0,
    "max_frames": 120,
}


@pytest.fixture
def vggt_omega(mock_vggt_omega_patches):  # noqa: ARG001 – patches applied via fixture
    """A fully-constructed :class:`VGGTOmegaReconstructor` with IO mocked out."""
    from src.reconstruct import VGGTOmegaReconstructor

    return VGGTOmegaReconstructor(**VGGT_OMEGA_KWARGS)
