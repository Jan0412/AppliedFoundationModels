"""VGGT-Omega reconstructor: RGB frames in, a :class:`FrameSet` out.

Loads VGGT-Omega from Hugging Face, predicts up-to-scale depth and camera
poses for a set of RGB frames, writes depth (confidence-filtered) and
confidence maps under ``out_dir``, and returns a back-projectable
:class:`~src.utils.datasets.FrameSet`.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

from src.utils.datasets import FrameSet

from .base import BaseReconstructor


def _average_intrinsics(intrinsics: np.ndarray) -> dict[str, float]:
    """Average per-frame ``(N, 3, 3)`` intrinsics into one calibration dict."""
    k = intrinsics.mean(axis=0)
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def _world_from_camera(extrinsic: np.ndarray) -> np.ndarray:
    """Invert a camera-from-world ``(3, 4)`` matrix to a ``(4, 4)`` cam-to-world."""
    t_cw = np.eye(4, dtype=np.float32)
    t_cw[:3, :4] = extrinsic
    return np.linalg.inv(t_cw).astype(np.float32)


def _save_depth_png(
    depth: np.ndarray,
    confidence: np.ndarray,
    path: Path,
    *,
    depth_scale: float,
    confidence_threshold: float,
) -> None:
    """Write predicted depth as a 16-bit PNG.

    Pixels with non-finite / non-positive depth, or confidence at or below
    *confidence_threshold*, are written as 0 (dropped by
    :func:`~src.utils.geometry.backproject`). Remaining values are stored as
    ``round(Z * depth_scale)`` so decode via ``raw / depth_scale`` recovers
    the up-to-scale prediction without uint16 truncating small floats.
    """
    valid = (
        np.isfinite(depth)
        & (depth > 0)
        & (confidence > confidence_threshold)
    )
    raw = np.zeros(depth.shape, dtype=np.float32)
    raw[valid] = depth[valid] * float(depth_scale)
    Image.fromarray(np.clip(np.rint(raw), 0, 65535).astype(np.uint16)).save(path)


def _sample_frame_paths(rgb_paths: list[str], max_frames: int) -> list[str]:
    """Pick up to *max_frames* paths evenly spaced across the sequence."""
    n = min(len(rgb_paths), max_frames)
    picks = np.linspace(0, len(rgb_paths) - 1, num=n, dtype=int)
    return [str(rgb_paths[i]) for i in picks]


class VGGTOmegaReconstructor(BaseReconstructor):
    """Reconstruct scene geometry from RGB frames with VGGT-Omega.

    All constructor arguments are required and map 1:1 to
    ``models.vggt_omega`` in ``config.yaml``.

    Args:
        model_id:              Hugging Face repository for the checkpoint.
        checkpoint:            Checkpoint filename inside *model_id*.
        device:                Torch device string (e.g. ``"cuda"``).
        image_resolution:      Longest-side / token budget passed to the
                               preprocessor (must be divisible by 16).
        preprocess_mode:       ``"balanced"`` or ``"max_size"`` (see
                               ``vggt_omega.utils.load_fn``).
        confidence_threshold:  Depth pixels at or below this ``depth_conf`` are
                               zeroed in the written PNGs.
        depth_scale:           PNG encode/decode divisor (``raw = Z * scale``).
                               Stored on the returned :class:`FrameSet`.
                               VGGT-Omega is up-to-scale; this is storage only.
        max_frames:            Evenly spaced subset size when the input sequence
                               is longer than this.
    """

    def __init__(
        self,
        model_id: str,
        checkpoint: str,
        device: str,
        image_resolution: int,
        preprocess_mode: str,
        confidence_threshold: float,
        depth_scale: float,
        max_frames: int,
    ) -> None:
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.image_resolution = image_resolution
        self.preprocess_mode = preprocess_mode
        self.confidence_threshold = confidence_threshold
        self.depth_scale = depth_scale
        self.max_frames = max_frames
        self._model = self._load_model()

    @classmethod
    def from_config(cls, path: str | Path = "config.yaml") -> "VGGTOmegaReconstructor":
        """Build from the required ``models.vggt_omega`` section of *path*."""
        cfg = yaml.safe_load(Path(path).read_text())
        return cls(**cfg["models"]["vggt_omega"])

    def _load_model(self) -> VGGTOmega:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        checkpoint_path = hf_hub_download(
            repo_id=self.model_id,
            filename=self.checkpoint,
        )

        model = VGGTOmega().to(self.device).eval()
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model

    def reconstruct(
        self,
        rgb_paths: list[str],
        out_dir: str | Path,
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> FrameSet:
        """Predict depth and poses for *rgb_paths* and return a :class:`FrameSet`.

        Writes under *out_dir*::

            depth/{i:05d}.png   # predicted depth (low-confidence pixels zeroed)
            conf/{i:05d}.npy    # full per-frame depth confidence

        Poses and averaged intrinsics live only in the returned :class:`FrameSet`
        (and later in LanceDB).
        """
        if not rgb_paths:
            raise ValueError("reconstruct() requires at least one RGB frame.")

        frame_paths = _sample_frame_paths(rgb_paths, self.max_frames)
        for path in frame_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"RGB frame not found: {path}")

        out_dir = Path(out_dir)
        depth_dir = out_dir / "depth"
        conf_dir = out_dir / "conf"
        depth_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        images = load_and_preprocess_images(
            frame_paths,
            mode=self.preprocess_mode,
            image_resolution=self.image_resolution,
        ).to(self.device)

        with torch.inference_mode():
            predictions = self._model(images)

        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        depths = predictions["depth"].squeeze(-1)[0].detach().float().cpu().numpy()
        confidences = predictions["depth_conf"][0].detach().float().cpu().numpy()
        extrinsics_np = extrinsics[0].detach().float().cpu().numpy()
        intrinsics_np = intrinsics[0].detach().float().cpu().numpy()

        del predictions, images, extrinsics, intrinsics
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        intrinsics_dict = _average_intrinsics(intrinsics_np)

        depth_paths: list[str] = []
        poses: list[np.ndarray] = []
        for i in range(len(frame_paths)):
            stem = f"{i:05d}"
            depth_path = depth_dir / f"{stem}.png"
            _save_depth_png(
                depths[i],
                confidences[i],
                depth_path,
                depth_scale=self.depth_scale,
                confidence_threshold=self.confidence_threshold,
            )
            depth_paths.append(str(depth_path))

            np.save(conf_dir / f"{stem}.npy", confidences[i].astype(np.float32))
            poses.append(_world_from_camera(extrinsics_np[i]))

            if on_progress is not None:
                on_progress(1)

        return FrameSet(
            paths=frame_paths,
            depth_paths=depth_paths,
            poses=poses,
            intrinsics=intrinsics_dict,
            depth_scale=self.depth_scale,
        )
