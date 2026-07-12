"""Tests for src.ingest.video (PyAV frame extraction)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import src.ingest.video as video_mod
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


# ---------------------------------------------------------------------------
# probe_frame_count against containers that under-declare themselves
#
# Streamed or remuxed videos often carry no frame count. The estimate only
# sizes a progress bar, so it must degrade rather than raise — faked here
# because PyAV always writes a frame count into a file it encodes itself.
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, frames=0, duration=None, time_base=None, average_rate=None):
        self.frames = frames
        self.duration = duration
        self.time_base = time_base
        self.average_rate = average_rate


class _FakeContainer:
    def __init__(self, stream):
        self.streams = SimpleNamespace(video=[stream])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def container_with(monkeypatch):
    """Make src.ingest.video.av.open yield a stream with the given metadata."""

    def _install(**stream_kw):
        stream = _FakeStream(**stream_kw)
        monkeypatch.setattr(
            video_mod.av, "open", lambda path: _FakeContainer(stream)
        )

    return _install


def test_probe_derives_the_count_from_the_duration(container_with):
    """No declared frame count, but duration × rate gives one: 2 s at 24 fps."""
    container_with(
        frames=0, duration=48, time_base=Fraction(1, 24), average_rate=24
    )

    assert probe_frame_count("any.mp4", fps=None) == 48
    assert probe_frame_count("any.mp4", fps=2.0) == 4      # 2 s sampled at 2 fps


def test_probe_returns_zero_when_the_container_declares_nothing(container_with):
    """Neither a frame count nor a duration — the caller clamps to max_frames."""
    container_with(frames=0, duration=None, average_rate=None)

    assert probe_frame_count("any.mp4", fps=2.0) == 0


def test_probe_returns_the_total_when_the_rate_is_unknown(container_with):
    """A frame count but no rate: sampling can't be estimated, so report them all."""
    container_with(frames=30, average_rate=None)

    assert probe_frame_count("any.mp4", fps=2.0) == 30
