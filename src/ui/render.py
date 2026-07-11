"""Drawing helpers for the viser app: scene, highlights, evidence, status text.

Pure presentation — nothing here touches the pipeline or the service, so the
formatting can be unit-tested without a browser or a GPU.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data_model import ProjectedObject, SearchState
from src.utils.geometry import aabb_corners

#: Per-object highlight colours, matching the notebooks' palette (first is red).
OBJ_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 0), (0, 162, 255), (255, 165, 0), (214, 51, 255),
    (0, 200, 83), (255, 224, 0), (141, 110, 99), (0, 191, 165),
]

#: Palette for 2D mask overlays in the evidence panel (as in the notebooks).
MASK_PALETTE: list[tuple[int, int, int]] = [
    (220, 50, 47), (38, 139, 210), (42, 161, 152), (133, 153, 0),
    (211, 54, 130), (181, 137, 0), (108, 113, 196), (203, 75, 22),
]
MASK_ALPHA = 110

#: The 12 edges of an axis-aligned box, indexing into aabb_corners' 8 points.
BOX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],     # bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],     # top face
    [0, 4], [1, 5], [2, 6], [3, 7],     # verticals
])

SCENE_NODE = "/scene"
HIGHLIGHT_NODE = "/highlight"


def object_color(index: int) -> tuple[int, int, int]:
    """Highlight colour for the *index*-th object of a result."""
    return OBJ_COLORS[index % len(OBJ_COLORS)]


def show_scene(server, points: np.ndarray, colors: Optional[np.ndarray], *, point_size: float):
    """Draw (or replace) the background scene cloud.

    Returns the point-cloud handle; re-adding under the same name replaces the
    previous cloud without disturbing the camera.
    """
    return server.scene.add_point_cloud(
        SCENE_NODE,
        points=points,
        colors=colors if colors is not None else (160, 160, 160),
        point_size=point_size,
        point_shape="rounded",
    )


def show_highlights(
    server,
    projected: list[ProjectedObject],
    *,
    point_size: float,
    query: str = "",
) -> list:
    """Mark each matched object: coloured points, a wireframe box, a label.

    Drawn as separate nodes over the untouched scene cloud, so clearing a query
    is a matter of removing them — the scene never has to be rebuilt.

    Returns the handles, for :func:`clear_highlights`.
    """
    handles = []
    for i, obj in enumerate(projected):
        color = object_color(i)

        handles.append(
            server.scene.add_point_cloud(
                f"{HIGHLIGHT_NODE}/{obj.id}",
                points=obj.points,
                colors=color,
                point_size=point_size * 1.5,
                point_shape="rounded",
            )
        )

        if obj.bbox is None:
            continue

        corners = aabb_corners(obj.bbox)
        handles.append(
            server.scene.add_line_segments(
                f"{HIGHLIGHT_NODE}/box-{obj.id}",
                points=corners[BOX_EDGES],          # (12, 2, 3)
                colors=color,
                line_width=3.0,
            )
        )
        top_centre = np.array([
            float(obj.bbox[:, 0].mean()),
            float(obj.bbox[:, 1].mean()),
            float(obj.bbox[1, 2]),
        ])
        handles.append(
            server.scene.add_label(
                f"{HIGHLIGHT_NODE}/label-{obj.id}",
                text=f"{query} ({obj.id})" if query else obj.id,
                position=top_centre,
            )
        )
    return handles


def clear_highlights(handles: list) -> None:
    """Remove previously drawn highlight nodes."""
    for handle in handles:
        handle.remove()


def results_markdown(state: SearchState) -> str:
    """Summarise a finished query: what was found, and how retrieval got there."""
    n_objects = len(state.projected or [])
    n_frames = len(state.results or [])

    if n_objects == 0:
        head = f"**No 3D object found** for _{state.query}_."
    else:
        noun = "object" if n_objects == 1 else "objects"
        head = (
            f"**{n_objects} {noun}** for _{state.query}_, "
            f"from {n_frames} supporting frame(s)."
        )

    lines = [head]

    diag = state.retrieval_diag
    if diag is not None:
        lines.append("")
        detail = [f"retrieval `{diag.mode}`", f"pool {diag.pool_size}"]
        if diag.total_frames:
            detail[-1] = f"pool {diag.pool_size}/{diag.total_frames}"
        if diag.n_selected is not None:
            detail.append(f"{diag.n_selected} diverse frames")
        if diag.separability is not None:
            detail.append(f"η {diag.separability:.2f}")
        lines.append(" · ".join(detail))
        if diag.gated:
            lines.append("")
            lines.append(
                "⚠️ Low separability — the scene probably does not contain this object."
            )

    return "\n".join(lines)


def evidence_caption(rank: int, hit) -> str:
    """One-line caption for a supporting frame in the evidence panel."""
    return (
        f"**#{rank}** · similarity {hit.similarity_score:.3f} · "
        f"detection {hit.detection_score:.3f}"
    )


def mask_overlay(hit, query: str, *, width: int = 320) -> np.ndarray:
    """Render a supporting frame with its SAM masks and boxes drawn on.

    This is the evidence behind a 3D highlight: the pixels that were
    back-projected into the object the user is looking at.

    Args:
        hit:   A :class:`~src.data_model.DetectedImage`.
        query: Query text, drawn next to each box.
        width: Output width; the frame is scaled down to keep the panel light.

    Returns:
        An ``(H, W, 3)`` uint8 RGB image.
    """
    canvas = Image.open(hit.path).convert("RGBA")

    for i, mask in enumerate(hit.masks or []):
        color = MASK_PALETTE[i % len(MASK_PALETTE)]
        mask_np = np.asarray(mask.cpu().numpy() if hasattr(mask, "cpu") else mask)
        overlay = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        overlay[mask_np > 0] = [*color, MASK_ALPHA]
        canvas = Image.alpha_composite(canvas, Image.fromarray(overlay, "RGBA"))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=14
        )
    except OSError:
        font = ImageFont.load_default()

    boxes = hit.boxes.tolist() if hasattr(hit.boxes, "tolist") else hit.boxes
    scores = hit.scores.tolist() if hasattr(hit.scores, "tolist") else hit.scores
    for i, (box, score) in enumerate(zip(boxes, scores)):
        color = MASK_PALETTE[i % len(MASK_PALETTE)]
        x0, y0, x1, y1 = (int(v) for v in box)
        draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=2)
        draw.text(
            (x0 + 3, max(y0 - 18, 0)),
            f"{query[:12]} {float(score):.2f}",
            fill=(255, 255, 255, 255),
            font=font,
        )

    canvas = canvas.convert("RGB")
    if canvas.width > width:
        height = round(canvas.height * width / canvas.width)
        canvas = canvas.resize((width, height), Image.BILINEAR)
    return np.asarray(canvas)
