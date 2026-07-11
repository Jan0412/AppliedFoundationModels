"""The viser app: browse a scene, query it, watch a video become one.

Run it::

    uv run python -m src.ui.app          # then open http://localhost:8080

Layout: the point cloud fills the window; the panel on the right picks a scene,
uploads a video, runs queries, and (folded away until asked for) shows the 2D
frames that produced each 3D highlight.

Every callback hands its work to a thread. viser's callbacks run on the
websocket handler, so a query or an ingest running inline would freeze the page
— including the progress bar meant to show that it hasn't.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import viser
import yaml

from . import render
from .service import SceneService

#: How often the ingest poller re-reads the job (seconds).
POLL_INTERVAL = 0.3

PROJECTION_MODES = ("cluster_single", "cluster_instances", "simple")


class SceneApp:
    """Wires viser widgets to :class:`SceneService`."""

    def __init__(self, service: SceneService, server: viser.ViserServer) -> None:
        self.service = service
        self.server = server

        self._highlights: list = []
        self._evidence: list = []
        # Bumped on every scene switch; a load that finishes after the user has
        # moved on sees a stale generation and drops its result.
        self._scene_generation = 0

        self._build_gui()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _build_gui(self) -> None:
        gui = self.server.gui
        scenes = self.service.list_scenes()

        with gui.add_folder("Scene"):
            self.scene_dropdown = gui.add_dropdown(
                "Collection",
                options=scenes or ("— none indexed —",),
                initial_value=scenes[0] if scenes else "— none indexed —",
            )
            self.upload_button = gui.add_upload_button(
                "Add scene from video",
                icon=viser.Icon.PLUS,
                mime_type="video/*",
            )
            self.progress = gui.add_progress_bar(0.0, visible=False, animated=True)
            self.scene_status = gui.add_markdown("")

        with gui.add_folder("Query"):
            self.query_input = gui.add_text("Text", initial_value="")
            self.mode_dropdown = gui.add_dropdown(
                "Projection", options=PROJECTION_MODES
            )
            self.search_button = gui.add_button("Search", icon=viser.Icon.SEARCH)
            self.clear_button = gui.add_button("Clear results")
            self.results_md = gui.add_markdown("")

        self.evidence_folder = gui.add_folder("Evidence", expand_by_default=False)

        self.scene_dropdown.on_update(lambda _: self._spawn(self._load_scene))
        self.upload_button.on_upload(lambda _: self._spawn(self._start_ingest))
        self.search_button.on_click(lambda _: self._spawn(self._search))
        self.clear_button.on_click(lambda _: self._clear_results())

        if scenes:
            self._spawn(self._load_scene)

    @staticmethod
    def _spawn(fn) -> None:
        threading.Thread(target=fn, daemon=True).start()

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _load_scene(self) -> None:
        collection_id = self.scene_dropdown.value
        if collection_id not in self.service.list_scenes():
            return

        self._scene_generation += 1
        generation = self._scene_generation

        self._clear_results()
        self.scene_status.content = f"Loading **{collection_id}** …"
        try:
            points, colors = self.service.load_scene(collection_id)
        except Exception as exc:
            self.scene_status.content = f"❌ Could not load **{collection_id}**: {exc}"
            return

        if generation != self._scene_generation:
            return          # the user has already picked a different scene

        render.show_scene(
            self.server, points, colors, point_size=self.service.voxel
        )
        self.scene_status.content = f"**{collection_id}** — {len(points):,} points"

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _search(self) -> None:
        query = self.query_input.value.strip()
        collection_id = self.scene_dropdown.value
        if not query or collection_id not in self.service.list_scenes():
            return

        self.search_button.disabled = True
        self.results_md.content = f"Searching for _{query}_ …"
        try:
            state = self.service.query(
                query, collection_id, mode=self.mode_dropdown.value
            )
        except Exception as exc:
            self.results_md.content = f"❌ Query failed: {exc}"
            return
        finally:
            self.search_button.disabled = False

        self._clear_results()
        self._highlights = render.show_highlights(
            self.server,
            state.projected or [],
            point_size=self.service.voxel,
            query=query,
        )
        self.results_md.content = render.results_markdown(
            state, detection_warn_threshold=self.service.detection_warn_threshold
        )
        self._show_evidence(state, query)

    def _show_evidence(self, state, query: str) -> None:
        """Fill the Evidence folder with the frames behind the 3D result."""
        with self.evidence_folder:
            for rank, hit in enumerate(state.results or [], start=1):
                self._evidence.append(
                    self.server.gui.add_markdown(render.evidence_caption(rank, hit))
                )
                try:
                    image = render.mask_overlay(hit, query)
                except (OSError, ValueError):
                    continue     # frame missing on disk — skip it, keep the rest
                self._evidence.append(
                    self.server.gui.add_image(image, label=Path(hit.path).name)
                )

    def _clear_results(self) -> None:
        render.clear_highlights(self._highlights)
        self._highlights = []
        for handle in self._evidence:
            handle.remove()
        self._evidence = []
        self.results_md.content = ""

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _start_ingest(self) -> None:
        upload = self.upload_button.value
        if upload is None:
            return

        self.upload_button.disabled = True
        self.progress.visible = True
        self.progress.value = 0.0
        self.scene_status.content = f"Ingesting **{upload.name}** …"

        job_id = self.service.ingest_video(upload.content, upload.name)
        self._poll_ingest(job_id)

    def _poll_ingest(self, job_id: str) -> None:
        """Mirror the job onto the progress bar until it finishes.

        The ingest thread only mutates its :class:`JobStatus`; reading it from
        here is what keeps :mod:`src.ingest` free of any UI dependency.
        """
        while True:
            status = self.service.job_status(job_id)
            if status is None:
                break

            self.progress.value = _percent(status)
            self.scene_status.content = _stage_line(status)

            if status["state"] in ("done", "failed"):
                break
            time.sleep(POLL_INTERVAL)

        self.progress.visible = False
        self.upload_button.disabled = False

        if status and status["state"] == "done":
            self.scene_dropdown.options = self.service.list_scenes()
            self.scene_dropdown.value = status["collection_id"]   # triggers the load


def _percent(status: dict) -> float:
    """Progress within the current stage, 0–100."""
    total = status.get("total") or 0
    if total <= 0:
        return 0.0
    return 100.0 * min(status["processed"] / total, 1.0)


def _stage_line(status: dict) -> str:
    """One line naming the stage and how far into it we are."""
    name = status["collection_id"]

    if status["state"] == "failed":
        return f"❌ **{name}** failed: {status['error']}"
    if status["state"] == "done":
        return f"✅ **{name}** is ready — {status['total']} frames indexed."

    stage, stages = status.get("stage"), status.get("stages") or []
    if not stage:
        return f"Preparing **{name}** …"

    step = stages.index(stage) + 1 if stage in stages else 1
    return (
        f"**{name}** — step {step}/{len(stages) or 1}: {stage} "
        f"({status['processed']}/{status['total']})"
    )


def main(config_path: str | Path = "config.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    port = (cfg.get("ui") or {}).get("port", 8080)

    service = SceneService(config_path)
    server = viser.ViserServer(port=port, label="3D Scene Search")
    SceneApp(service, server)

    # Pay for SigLIP + SAM now, off the request path, so the first query is a
    # query rather than a model download.
    threading.Thread(target=service.warmup, daemon=True).start()

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
