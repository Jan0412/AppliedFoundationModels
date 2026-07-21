"""Tests for src.reconstruct.vggt_omega.

VGGT-Omega itself is stubbed: these pin the FrameSet contract, depth PNG
encoding, frame sampling, and config wiring — the same surface the ingest
pipeline depends on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.reconstruct import BaseReconstructor, VGGTOmegaReconstructor
from src.reconstruct.vggt_omega import (
    _average_intrinsics,
    _sample_frame_paths,
    _save_depth_png,
    _world_from_camera,
)
from src.utils.datasets import FrameSet
from src.utils.geometry import build_scene_cloud

from .conftest import VGGT_OMEGA_KWARGS


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_sample_frame_paths_spreads_across_the_sequence():
    paths = [f"{i}.jpg" for i in range(10)]
    assert _sample_frame_paths(paths, max_frames=3) == ["0.jpg", "4.jpg", "9.jpg"]


def test_sample_frame_paths_clamps_to_available_frames():
    paths = ["a.jpg", "b.jpg"]
    assert _sample_frame_paths(paths, max_frames=50) == paths


def test_average_intrinsics_means_per_frame_ks():
    ks = np.array(
        [
            [[10.0, 0.0, 1.0], [0.0, 20.0, 2.0], [0.0, 0.0, 1.0]],
            [[30.0, 0.0, 5.0], [0.0, 40.0, 6.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    assert _average_intrinsics(ks) == {"fx": 20.0, "fy": 30.0, "cx": 3.0, "cy": 4.0}


def test_world_from_camera_inverts_a_translation():
    # Camera-from-world: identity R, t = (2, 0, 0) → cam at world (+2, 0, 0)
    # after inversion of the 4x4.
    extri = np.zeros((3, 4), dtype=np.float32)
    extri[:3, :3] = np.eye(3, dtype=np.float32)
    extri[0, 3] = 2.0

    c2w = _world_from_camera(extri)

    assert c2w.shape == (4, 4)
    np.testing.assert_allclose(c2w[:3, :3], np.eye(3), atol=1e-5)
    np.testing.assert_allclose(c2w[:3, 3], [-2.0, 0.0, 0.0], atol=1e-5)


def test_save_depth_png_zeros_low_confidence_and_invalid(tmp_path):
    depth = np.array([[1.0, 2.0], [np.nan, -1.0]], dtype=np.float32)
    conf = np.array([[100.0, 1.0], [100.0, 100.0]], dtype=np.float32)
    path = tmp_path / "d.png"

    _save_depth_png(
        depth, conf, path, depth_scale=1000.0, confidence_threshold=10.0
    )

    raw = np.asarray(Image.open(path))
    assert raw.dtype == np.uint16
    assert raw[0, 0] == 1000          # 1.0 m * 1000
    assert raw[0, 1] == 0             # conf below threshold
    assert raw[1, 0] == 0             # nan
    assert raw[1, 1] == 0             # non-positive


# ---------------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------------


def test_implements_the_reconstructor_contract(mock_vggt_omega_patches):  # noqa: ARG001
    assert issubclass(VGGTOmegaReconstructor, BaseReconstructor)


def test_from_config_reads_vggt_omega_section(tmp_path, mock_vggt_omega_patches):  # noqa: ARG001
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"models": {"vggt_omega": VGGT_OMEGA_KWARGS}}))

    rec = VGGTOmegaReconstructor.from_config(config)

    assert rec.model_id == VGGT_OMEGA_KWARGS["model_id"]
    assert rec.max_frames == VGGT_OMEGA_KWARGS["max_frames"]
    assert rec.device.type == "cpu"


def test_from_config_missing_section_raises(tmp_path, mock_vggt_omega_patches):  # noqa: ARG001
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"models": {}}))

    with pytest.raises(KeyError):
        VGGTOmegaReconstructor.from_config(config)


def test_init_downloads_and_loads_the_checkpoint(mock_vggt_omega_patches):
    VGGTOmegaReconstructor(**VGGT_OMEGA_KWARGS)

    mock_vggt_omega_patches["VGGTOmega"].assert_called_once_with()
    mock_vggt_omega_patches["model"].load_state_dict.assert_called_once_with({})
    mock_vggt_omega_patches["model"].eval.assert_called_once()


# ---------------------------------------------------------------------------
# reconstruct()
# ---------------------------------------------------------------------------


def test_returns_a_frameset_of_the_requested_length(vggt_omega, rgb_frames, tmp_path):
    paths = rgb_frames(n=3)

    fs = vggt_omega.reconstruct(paths, tmp_path / "out")

    assert isinstance(fs, FrameSet)
    assert len(fs) == 3
    assert len(fs.depth_paths) == 3 and len(fs.poses) == 3


def test_output_is_usable_downstream(vggt_omega, rgb_frames, tmp_path):
    paths = rgb_frames(n=4)
    fs = vggt_omega.reconstruct(paths, tmp_path / "out")

    assert all(Path(p).is_file() for p in fs.paths)
    assert all(Path(p).is_file() for p in fs.depth_paths)
    for pose in fs.poses:
        assert pose.shape == (4, 4)
        assert np.isfinite(pose).all()
    assert set(fs.intrinsics) == {"fx", "fy", "cx", "cy"}
    assert fs.depth_scale == VGGT_OMEGA_KWARGS["depth_scale"]

    assert np.asarray(Image.open(fs.depth_paths[0])).dtype == np.uint16

    points, _ = build_scene_cloud(fs, n_frames=4, voxel=0.05)
    assert len(points) > 0


def test_writes_depth_and_confidence_under_out_dir(vggt_omega, rgb_frames, tmp_path):
    out = tmp_path / "out"
    fs = vggt_omega.reconstruct(rgb_frames(n=2), out)

    assert (out / "depth" / "00000.png").is_file()
    assert (out / "depth" / "00001.png").is_file()
    assert (out / "conf" / "00000.npy").is_file()
    assert (out / "conf" / "00001.npy").is_file()
    assert fs.depth_paths == [
        str(out / "depth" / "00000.png"),
        str(out / "depth" / "00001.png"),
    ]


def test_depth_png_zeros_the_low_confidence_pixel(vggt_omega, rgb_frames, tmp_path):
    fs = vggt_omega.reconstruct(rgb_frames(n=1), tmp_path / "out")
    raw = np.asarray(Image.open(fs.depth_paths[0]))

    assert raw[0, 0] == 0                         # conf=1.0 in the fake predictions
    assert raw[0, 1] == 2000                      # 2.0 m * 1000


def test_samples_when_input_exceeds_max_frames(
    mock_vggt_omega_patches, rgb_frames, tmp_path  # noqa: ARG001
):
    kwargs = {**VGGT_OMEGA_KWARGS, "max_frames": 3}
    rec = VGGTOmegaReconstructor(**kwargs)
    paths = rgb_frames(n=10)

    fs = rec.reconstruct(paths, tmp_path / "out")

    assert len(fs) == 3
    assert fs.paths == [paths[0], paths[4], paths[9]]


def test_reports_progress_once_per_output_frame(vggt_omega, rgb_frames, tmp_path):
    ticks: list[int] = []

    fs = vggt_omega.reconstruct(
        rgb_frames(n=4), tmp_path / "out", on_progress=ticks.append
    )

    assert sum(ticks) == len(fs) == 4


def test_empty_rgb_paths_raises(vggt_omega, tmp_path):
    with pytest.raises(ValueError, match="at least one RGB frame"):
        vggt_omega.reconstruct([], tmp_path / "out")


def test_missing_rgb_file_raises(vggt_omega, tmp_path):
    with pytest.raises(FileNotFoundError, match="RGB frame not found"):
        vggt_omega.reconstruct([str(tmp_path / "gone.jpg")], tmp_path / "out")
