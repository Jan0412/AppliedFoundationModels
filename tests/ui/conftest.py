"""Shared fixtures for tests/ui/.

populated_store – a tmp LanceDB with one small, fully-projectable collection
                  plus its calibration row (the shape Indexer writes), so cache
                  and service tests run against a real store with no models.
fake_server     – a stand-in for viser.ViserServer, so the app and the render
                  helpers can be driven without a browser or a websocket.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from src.utils.db import connect

EMBED_DIM = 8


# ---------------------------------------------------------------------------
# Fake viser server
# ---------------------------------------------------------------------------


class FakeHandle:
    """A viser handle: attributes are plain state, callbacks are recorded.

    viser hands back one of these from every ``add_*`` call; the app then reads
    and writes ``.value`` / ``.content`` / ``.disabled`` on it, and registers
    callbacks. Recording all of it lets a test assert on what the user would see.
    """

    def __init__(self, kind: str, **props) -> None:
        self.kind = kind
        self.__dict__.update(props)
        self.disabled = False
        self.removed = False
        self.callbacks: dict[str, list] = {}

    def _register(self, event: str, fn):
        self.callbacks.setdefault(event, []).append(fn)
        return fn

    def on_update(self, fn):
        return self._register("update", fn)

    def on_click(self, fn):
        return self._register("click", fn)

    def on_upload(self, fn):
        return self._register("upload", fn)

    def fire(self, event: str = "update") -> None:
        """Simulate the user touching this widget."""
        for fn in self.callbacks.get(event, []):
            fn(self)

    def remove(self) -> None:
        self.removed = True


class FakeFolder(FakeHandle):
    """``gui.add_folder`` returns a context manager; nesting is a no-op here."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeGui:
    """The ``server.gui`` namespace — every widget the app builds."""

    def __init__(self) -> None:
        self.handles: list[FakeHandle] = []

    def _add(self, kind: str, **props) -> FakeHandle:
        handle = FakeHandle(kind, **props)
        self.handles.append(handle)
        return handle

    def add_folder(self, label, expand_by_default=True) -> FakeFolder:
        folder = FakeFolder("folder", label=label, expand_by_default=expand_by_default)
        self.handles.append(folder)
        return folder

    def add_dropdown(self, label, options=(), initial_value=None, hint=None):
        return self._add(
            "dropdown", label=label, options=tuple(options),
            value=initial_value, hint=hint,
        )

    def add_slider(self, label, min=None, max=None, step=None, initial_value=None, hint=None):
        return self._add(
            "slider", label=label, min=min, max=max, step=step,
            value=initial_value, hint=hint,
        )

    def add_text(self, label, initial_value=""):
        return self._add("text", label=label, value=initial_value)

    def add_button(self, label, icon=None):
        return self._add("button", label=label, icon=icon)

    def add_upload_button(self, label, icon=None, mime_type=None):
        return self._add(
            "upload", label=label, icon=icon, mime_type=mime_type, value=None
        )

    def add_progress_bar(self, value=0.0, visible=True, animated=False):
        return self._add(
            "progress", value=value, visible=visible, animated=animated
        )

    def add_markdown(self, content=""):
        return self._add("markdown", content=content)

    def add_image(self, image, label=None):
        return self._add("image", image=image, label=label)


class FakeScene:
    """The ``server.scene`` namespace — the 3D nodes the render helpers add."""

    def __init__(self) -> None:
        self.nodes: list[FakeHandle] = []

    def _add(self, kind: str, name: str, **props) -> FakeHandle:
        node = FakeHandle(kind, name=name, **props)
        self.nodes.append(node)
        return node

    def add_point_cloud(self, name, points, colors, point_size, point_shape=None):
        return self._add(
            "point_cloud", name, points=points, colors=colors,
            point_size=point_size, point_shape=point_shape,
        )

    def add_line_segments(self, name, points, colors, line_width):
        return self._add(
            "line_segments", name, points=points, colors=colors,
            line_width=line_width,
        )

    def add_label(self, name, text, position):
        return self._add("label", name, text=text, position=position)

    def names_of(self, kind: str) -> list[str]:
        return [n.name for n in self.nodes if n.kind == kind]


class FakeServer:
    """Enough of ``viser.ViserServer`` for the app and the render helpers."""

    def __init__(self) -> None:
        self.gui = FakeGui()
        self.scene = FakeScene()


@pytest.fixture
def fake_server() -> FakeServer:
    return FakeServer()


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
