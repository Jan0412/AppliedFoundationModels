"""Video → RGB frames on disk.

Frames must end up as files, not in memory: the indexer stores each frame's
path, and retrieval lazy-loads images by that path at query time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import av


def probe_frame_count(video_path: str | Path, fps: float | None = None) -> int:
    """Estimate how many frames :func:`extract_frames` will keep.

    Only used to give the extract stage a progress total, so an estimate is
    fine — containers that declare neither a frame count nor a duration fall
    back to 0 and the caller clamps.

    Args:
        video_path: The video to probe.
        fps:        Sampling rate that will be used, or ``None`` for every frame.

    Returns:
        The estimated number of sampled frames (0 when unknown).
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]

        total = stream.frames or 0
        if not total and stream.duration and stream.average_rate:
            total = int(float(stream.duration * stream.time_base) * float(stream.average_rate))
        if not total:
            return 0

        if fps is None or not stream.average_rate:
            return total

        source_fps = float(stream.average_rate)
        duration_s = total / source_fps if source_fps else 0.0
        return max(1, int(duration_s * fps))


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    fps: float | None = 2.0,
    max_frames: int = 300,
    on_progress: Callable[[int], None] | None = None,
) -> list[str]:
    """Decode *video_path* and write sampled RGB frames into *out_dir*.

    Sampling at a few frames per second is deliberate: consecutive video frames
    are near-duplicates that cost embedding time and add nothing to retrieval.

    Args:
        video_path:  Video file to decode.
        out_dir:     Directory for the extracted JPEGs (created if absent).
        fps:         Frames to keep per second of video; ``None`` keeps all.
        max_frames:  Stop after this many frames.
        on_progress: Called with 1 per frame written.

    Returns:
        Paths of the written frames, in capture order. Names are zero-padded
        and sequential, so re-ingesting the same video derives the same row ids.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    interval = 1.0 / fps if fps else 0.0
    paths: list[str] = []
    next_time = 0.0

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        for frame in container.decode(stream):
            if len(paths) >= max_frames:
                break

            # frame.time is None for containers without timestamps — keep those
            # frames rather than dropping the whole video.
            t = frame.time
            if interval and t is not None:
                if t + 1e-6 < next_time:
                    continue
                next_time = t + interval

            path = out_dir / f"{len(paths):06d}.jpg"
            frame.to_image().save(path, quality=95)
            paths.append(str(path))
            if on_progress is not None:
                on_progress(1)

    return paths
