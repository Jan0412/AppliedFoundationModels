"""Tests for src.ingest.video (PyAV frame extraction)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.ingest import extract_frames, probe_frame_count


def test_extract_writes_frames_to_disk(tiny_video, tmp_path):
    """Frames must be files — the indexer stores paths and reloads them later."""
    video = tiny_video(n_frames=16, fps=8)
    out = tmp_path / "rgb"

    paths = extract_frames(video, out, fps=None)

    assert len(paths) == 16
    assert all(Path(p).is_file() for p in paths)
    assert Image.open(paths[0]).size == (64, 64)


def test_extract_samples_at_the_requested_fps(tiny_video, tmp_path):
    """A 2-second 8fps clip sampled at 2fps yields ~4 frames, not 16."""
    video = tiny_video(n_frames=16, fps=8)
    paths = extract_frames(video, tmp_path / "rgb", fps=2.0)
    assert 3 <= len(paths) <= 5


def test_extract_respects_max_frames(tiny_video, tmp_path):
    video = tiny_video(n_frames=16, fps=8)
    paths = extract_frames(video, tmp_path / "rgb", fps=None, max_frames=5)
    assert len(paths) == 5


def test_frame_names_are_sequential_and_deterministic(tiny_video, tmp_path):
    """Stable names → stable derived row ids when a video is re-ingested."""
    video = tiny_video(n_frames=4, fps=4)
    first = extract_frames(video, tmp_path / "a", fps=None)
    second = extract_frames(video, tmp_path / "b", fps=None)

    assert [Path(p).name for p in first] == ["000000.jpg", "000001.jpg",
                                             "000002.jpg", "000003.jpg"]
    assert [Path(p).name for p in first] == [Path(p).name for p in second]


def test_extract_reports_progress_per_frame(tiny_video, tmp_path):
    video = tiny_video(n_frames=8, fps=8)
    ticks: list[int] = []

    paths = extract_frames(video, tmp_path / "rgb", fps=None, on_progress=ticks.append)

    assert sum(ticks) == len(paths)


def test_extract_creates_the_output_dir(tiny_video, tmp_path):
    video = tiny_video(n_frames=2, fps=2)
    out = tmp_path / "deep" / "nested" / "rgb"
    extract_frames(video, out, fps=None)
    assert out.is_dir()


def test_probe_estimates_the_sampled_frame_count(tiny_video):
    video = tiny_video(n_frames=16, fps=8)
    assert probe_frame_count(video, fps=None) >= 16
    assert 1 <= probe_frame_count(video, fps=2.0) <= 6
