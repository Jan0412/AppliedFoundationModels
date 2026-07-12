"""Tests for src.ui.app (SceneApp) against a fake viser server.

The app is wiring: widgets in, service calls out, results back onto widgets.
So the service is faked (no models, no DB — those are covered in tests/ui/
test_service.py and tests/query/) and the viser server is faked (see the
``fake_server`` fixture), leaving the wiring itself as the thing under test.

Callbacks normally run on a thread; ``inline_spawn`` runs them in the caller so
a test can assert straight after triggering one.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.ui.app as app_mod
from src.data_model import DetectedImage, ProjectedObject, SearchState
from src.ui import settings as settings_spec
from src.ui.app import SceneApp, _percent, _stage_line
from tests.ui.conftest import FakeServer


class FakeService:
    """SceneService's interface, with none of its cost."""

    def __init__(self, scenes=("room",)):
        self._scenes = list(scenes)
        self.voxel = 0.02
        self.detection_warn_threshold = 0.6
        self._settings = settings_spec.defaults_from_config({})

        self.points = np.zeros((7, 3), dtype=np.float32)
        self.colors = None
        self.state = SearchState(query="chair", collection_id="room")

        # Failures a test can arm, and calls a test can assert on.
        self.load_error: Exception | None = None
        self.query_error: Exception | None = None
        self.on_load = None
        self.queries: list[tuple[str, str]] = []
        self.uploads: list[tuple[bytes, str]] = []
        self.status_seq: list[dict | None] = []

    def list_scenes(self):
        return list(self._scenes)

    def load_scene(self, collection_id):
        if self.on_load is not None:
            self.on_load()
        if self.load_error is not None:
            raise self.load_error
        return self.points, self.colors

    def settings(self):
        return dict(self._settings)

    def update_settings(self, values):
        self._settings.update(values)

    def reset_settings(self):
        self._settings = settings_spec.defaults_from_config({})
        return dict(self._settings)

    def query(self, text, collection_id):
        self.queries.append((text, collection_id))
        if self.query_error is not None:
            raise self.query_error
        return self.state

    def ingest_video(self, content, filename):
        self.uploads.append((content, filename))
        return "job-1"

    def job_status(self, job_id):
        return self.status_seq.pop(0) if self.status_seq else None


@pytest.fixture
def inline_spawn(monkeypatch):
    """Run the callbacks' work in the caller instead of on a daemon thread."""
    monkeypatch.setattr(SceneApp, "_spawn", staticmethod(lambda fn: fn()))
    monkeypatch.setattr(app_mod, "POLL_INTERVAL", 0.0)


@pytest.fixture
def service():
    return FakeService()


@pytest.fixture
def app(service, fake_server, inline_spawn):
    return SceneApp(service, fake_server)


def _projected(obj_id="obj-0", n_points=5, bbox=True):
    return ProjectedObject(
        id=obj_id, path="",
        points=np.zeros((n_points, 3), dtype=np.float32), colors=None,
        bbox=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32) if bbox else None,
    )


def _hit(path="p.png", detection_score=0.9):
    return DetectedImage(
        id="x", path=path, similarity_score=0.5,
        detection_score=detection_score,
        boxes=torch.zeros((0, 4)), scores=torch.zeros((0,)), masks=[],
    )


def _status(state="running", **kw):
    base = dict(
        job_id="job-1", collection_id="clip", state=state,
        processed=0, total=4, stage="extract",
        stages=["extract", "reconstruct", "index"], error=None,
    )
    base.update(kw)
    return base


def test_spawn_runs_the_work_off_the_callback_thread():
    """The real _spawn (the other tests stub it out): viser's callbacks run on
    the websocket handler, so anything slow must leave it."""
    import threading

    done = threading.Event()
    caller = threading.current_thread().name
    ran_on: list[str] = []

    def _work():
        ran_on.append(threading.current_thread().name)
        done.set()

    SceneApp._spawn(_work)

    assert done.wait(timeout=5.0)
    assert ran_on[0] != caller


# ---------------------------------------------------------------------------
# Construction + scene loading
# ---------------------------------------------------------------------------


def test_building_the_gui_loads_the_first_scene(app, service, fake_server):
    """A page load lands on a drawn scene, not an empty viewport."""
    assert app.scene_dropdown.value == "room"
    assert fake_server.scene.names_of("point_cloud") == ["/scene"]
    assert "room" in app.scene_status.content
    assert "7 points" in app.scene_status.content


def test_no_indexed_scenes_shows_a_placeholder(fake_server, inline_spawn):
    app = SceneApp(FakeService(scenes=()), fake_server)

    assert app.scene_dropdown.value == "— none indexed —"
    assert fake_server.scene.nodes == []      # nothing to draw, nothing drawn


def test_switching_scene_redraws_under_the_same_node(app, service, fake_server):
    """Re-adding /scene replaces the cloud, so the camera is not disturbed."""
    service._scenes.append("kitchen")
    app.scene_dropdown.value = "kitchen"
    app.scene_dropdown.fire("update")

    assert fake_server.scene.names_of("point_cloud") == ["/scene", "/scene"]
    assert "kitchen" in app.scene_status.content


def test_unknown_collection_is_not_loaded(app, fake_server):
    before = len(fake_server.scene.nodes)
    app.scene_dropdown.value = "does-not-exist"
    app.scene_dropdown.fire("update")

    assert len(fake_server.scene.nodes) == before


def test_failed_scene_load_is_surfaced_not_raised(app, service, fake_server):
    service.load_error = RuntimeError("corrupt table")
    before = len(fake_server.scene.nodes)

    app._load_scene()

    assert "❌" in app.scene_status.content
    assert "corrupt table" in app.scene_status.content
    assert len(fake_server.scene.nodes) == before


def test_a_stale_scene_load_drops_its_result(app, service, fake_server):
    """The user switched scene mid-load: the older load must not draw over the newer.

    Simulated by bumping the generation from inside load_scene — exactly what a
    second _load_scene on another thread would have done.
    """
    before = len(fake_server.scene.nodes)
    service.on_load = lambda: setattr(
        app, "_scene_generation", app._scene_generation + 1
    )

    app._load_scene()

    assert len(fake_server.scene.nodes) == before      # result dropped
    assert app.scene_status.content.startswith("Loading")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_search_highlights_the_objects_and_reports_them(app, service, fake_server):
    service.state = SearchState(
        query="chair", collection_id="room",
        projected=[_projected("obj-0"), _projected("obj-1")],
        results=[_hit()],
    )
    app.query_input.value = "chair"
    app.search_button.fire("click")

    assert service.queries == [("chair", "room")]
    # One cloud + one box + one label per object, over the untouched scene.
    assert fake_server.scene.names_of("point_cloud") == [
        "/scene", "/highlight/obj-0", "/highlight/obj-1"
    ]
    assert len(app._highlights) == 6
    assert "2 objects" in app.results_md.content
    assert app.search_button.disabled is False


def test_search_ignores_a_blank_query(app, service):
    app.query_input.value = "   "
    app.search_button.fire("click")

    assert service.queries == []


def test_failed_query_is_surfaced_and_reenables_the_button(app, service):
    service.query_error = RuntimeError("CUDA OOM")
    app.query_input.value = "chair"

    app.search_button.fire("click")

    assert "❌" in app.results_md.content
    assert "CUDA OOM" in app.results_md.content
    # The finally clause matters: a failed query must not leave a dead button.
    assert app.search_button.disabled is False


def test_evidence_pairs_a_caption_with_each_frame(app, service, tmp_path):
    from PIL import Image
    frame = tmp_path / "f.png"
    Image.new("RGB", (16, 12), color=(30, 30, 30)).save(frame)

    service.state = SearchState(
        query="chair", collection_id="room",
        projected=[_projected()], results=[_hit(path=str(frame))],
    )
    app.query_input.value = "chair"
    app.search_button.fire("click")

    kinds = [h.kind for h in app._evidence]
    assert kinds == ["markdown", "image"]
    assert app._evidence[1].label == "f.png"


def test_evidence_skips_frames_missing_from_disk(app, service):
    """A deleted frame loses its thumbnail, not the whole panel."""
    service.state = SearchState(
        query="chair", collection_id="room",
        projected=[_projected()], results=[_hit(path="/nonexistent/gone.png")],
    )
    app.query_input.value = "chair"
    app.search_button.fire("click")

    assert [h.kind for h in app._evidence] == ["markdown"]   # caption kept
    assert "1 object" in app.results_md.content              # rest still rendered


def test_clear_results_removes_highlights_and_evidence(app, service, tmp_path):
    from PIL import Image
    frame = tmp_path / "f.png"
    Image.new("RGB", (16, 12), color=(30, 30, 30)).save(frame)
    service.state = SearchState(
        query="chair", collection_id="room",
        projected=[_projected()], results=[_hit(path=str(frame))],
    )
    app.query_input.value = "chair"
    app.search_button.fire("click")
    highlights, evidence = list(app._highlights), list(app._evidence)

    app.clear_button.fire("click")

    assert all(h.removed for h in highlights)
    assert all(h.removed for h in evidence)
    assert app._highlights == [] and app._evidence == []
    assert app.results_md.content == ""


# ---------------------------------------------------------------------------
# Configuration panel
# ---------------------------------------------------------------------------


def test_every_field_gets_a_widget_seeded_from_the_service(app, service):
    assert set(app.config_widgets) == {f.key for f in settings_spec.FIELDS}
    assert app.config_widgets["mode"].kind == "dropdown"      # choice field
    assert app.config_widgets["min_views"].kind == "slider"   # numeric field
    assert app.config_widgets["min_views"].value == service.settings()["min_views"]


def test_moving_a_widget_stages_the_value_on_the_service(app, service):
    widget = app.config_widgets["min_views"]
    widget.value = 4
    widget.fire("update")

    assert service.settings()["min_views"] == 4


def test_reset_pushes_the_config_values_back_onto_the_widgets(app, service):
    widget = app.config_widgets["min_views"]
    widget.value = 9
    widget.fire("update")

    app.reset_button.fire("click")

    assert widget.value == settings_spec.defaults_from_config({})["min_views"]
    assert service.settings()["min_views"] == widget.value


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def test_upload_runs_the_job_and_switches_to_the_new_scene(app, service, fake_server):
    service.status_seq = [
        _status(processed=2),
        _status(state="done", stage="index", processed=4, collection_id="clip"),
    ]
    app.upload_button.value = SimpleNamespace(name="clip.mp4", content=b"bytes")

    app.upload_button.fire("upload")

    assert service.uploads == [(b"bytes", "clip.mp4")]
    # The poller left the bar hidden and the button live again…
    assert app.progress.visible is False
    assert app.upload_button.disabled is False
    # …and the finished scene is selected, which draws it.
    assert app.scene_dropdown.value == "clip"
    assert "clip" in app.scene_status.content


def test_a_failed_job_leaves_the_ui_usable(app, service):
    service.status_seq = [_status(state="failed", error="no frames decoded")]
    app.upload_button.value = SimpleNamespace(name="clip.mp4", content=b"bytes")

    app.upload_button.fire("upload")

    assert "❌" in app.scene_status.content
    assert "no frames decoded" in app.scene_status.content
    assert app.progress.visible is False
    assert app.upload_button.disabled is False
    assert app.scene_dropdown.value == "room"      # scene unchanged


def test_a_vanished_job_stops_the_poller(app, service):
    """job_status returning None (unknown id) must not spin forever."""
    service.status_seq = []
    app.upload_button.value = SimpleNamespace(name="clip.mp4", content=b"bytes")

    app.upload_button.fire("upload")

    assert app.progress.visible is False
    assert app.upload_button.disabled is False


def test_upload_without_a_file_is_a_no_op(app, service):
    app.upload_button.value = None
    app.upload_button.fire("upload")

    assert service.uploads == []
    assert app.progress.visible is False


# ---------------------------------------------------------------------------
# Progress formatting
# ---------------------------------------------------------------------------


def test_percent_is_the_share_of_the_current_stage():
    assert _percent(_status(processed=1, total=4)) == 25.0
    assert _percent(_status(processed=9, total=4)) == 100.0   # clamped
    assert _percent(_status(processed=0, total=0)) == 0.0     # unknown total


def test_stage_line_numbers_the_stage_within_the_plan():
    line = _stage_line(_status(stage="reconstruct", processed=3, total=4))
    assert "step 2/3" in line
    assert "reconstruct" in line
    assert "3/4" in line


def test_stage_line_reports_terminal_states():
    done = _stage_line(_status(state="done", processed=4, total=4))
    assert "✅" in done and "4 frames" in done

    failed = _stage_line(_status(state="failed", error="boom"))
    assert "❌" in failed and "boom" in failed


def test_stage_line_before_the_first_stage_is_entered():
    assert "Preparing" in _stage_line(_status(stage=None))


def test_stage_line_tolerates_a_stage_outside_the_plan():
    line = _stage_line(_status(stage="mystery", stages=[]))
    assert "step 1/1" in line


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def test_main_serves_on_the_configured_port_and_warms_up(tmp_path, monkeypatch):
    """main() wires service → server → app, and pays for the models off-path."""
    import yaml

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"ui": {"port": 9999}}))

    built: dict = {}
    warmed: list[bool] = []

    class _Service:
        def __init__(self, path):
            built["config_path"] = path

        def warmup(self):
            warmed.append(True)

    class _InlineThread:
        """Runs the target on start(), so warmup is observable without a join."""

        def __init__(self, target, daemon=False):
            self._target = target

        def start(self):
            self._target()

    def _serve(port, label):
        built["port"] = port
        return FakeServer()

    def _sleep_forever(_seconds):
        raise KeyboardInterrupt      # break out of `while True: sleep(3600)`

    monkeypatch.setattr(app_mod, "SceneService", _Service)
    monkeypatch.setattr(app_mod.viser, "ViserServer", _serve)
    monkeypatch.setattr(app_mod, "SceneApp", lambda service, server: built.setdefault("app", (service, server)))
    monkeypatch.setattr(app_mod.threading, "Thread", _InlineThread)
    monkeypatch.setattr(app_mod.time, "sleep", _sleep_forever)

    with pytest.raises(KeyboardInterrupt):
        app_mod.main(config)

    assert built["port"] == 9999
    assert built["config_path"] == config
    assert warmed == [True]
