#!/usr/bin/env python3
"""ScanRefer benchmark (unique-object subset) for the Search2D pipeline.

Runs the pipeline on ScanRefer val scenes, restricted to objects whose category
appears exactly once in the scene ("unique" objects). Reports Acc@0.25 / Acc@0.5
for each 3D projection mode (``simple`` | ``cluster_single`` |
``cluster_instances``).

All modes are scored in a single pass: the 2D half of the pipeline
(embed | retrieve | select | detect | rerank) is mode-independent, so it runs
**once** per item and only the projection step is repeated per mode. The modes
therefore see byte-identical 2D results — differences in accuracy come from the
projection alone — and the run costs barely more than a single-mode one.

The embedder (``models.embedder``: ``siglip`` | ``clip``), retrieval and
projection parameters all come from ``config.yaml`` (the CONFIG block below only
holds eval-specific overrides), so tuning the config changes what is measured.
Each embedder indexes into its own LanceDB store and writes its own results
files, so a SigLIP run and a CLIP run never overwrite each other.

Usage
-----
Edit the CONFIG block below, then:

    cd /path/to/AppliedFoundationModels
    uv run python scripts/eval_scanrefer_unique.py

    # CLIP instead of the configured embedder:
    AFM_EMBEDDER=clip uv run python scripts/eval_scanrefer_unique.py

    # single projection mode:
    EVAL_PROJECTION_MODES=cluster_instances uv run python scripts/eval_scanrefer_unique.py

    # full-sentence queries instead of the category noun:
    EVAL_QUERY_MODE=description uv run python scripts/eval_scanrefer_unique.py

    # smoke test on two scenes before the full run:
    EVAL_N_SCENES=2 uv run python scripts/eval_scanrefer_unique.py

Re-runs are cheap: scenes already in that embedder's LanceDB are skipped at the
indexing step. Switching embedder means a full re-index (~234k ScanNet frames
for the 141-scene val subset), because the vectors are not interchangeable.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import yaml

# ── ensure project root on path ─────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these before running
# ════════════════════════════════════════════════════════════════════════════

SCANS_ROOT    = Path("/storage/group/dataset_mirrors/scannet/scans")
SCANREFER_DIR = Path("/usr/prakt/s0036/scanrefer")
CONFIG_YAML   = _ROOT / "config.yaml"

# Loaded once so the defaults below track config.yaml instead of duplicating
# its values as separate literals.
_CFG  = yaml.safe_load(CONFIG_YAML.read_text())
_QCFG = _CFG.get("query") or {}


# Subset: either set SCENES to an explicit list of scene ids, or leave it None
# and set N_SCENES to use the first N scenes from ScanRefer_filtered_val.txt.
# EVAL_N_SCENES trims the subset from the env — a 1-2 scene run is the cheap
# smoke test before committing to the full 141.
SCENES: list[str] | None = None
N_SCENES: int = int(os.environ.get("EVAL_N_SCENES", "141"))

# Query mode:
#   "object_name"  — one query per unique (scene, object_id);  query = category noun.
#   "description"  — one query per val description of a unique object; query = sentence.
QUERY_MODE: str = os.environ.get("EVAL_QUERY_MODE", "object_name")

# Short tag for the results filename. The two query modes score different item
# sets (722 unique objects vs 3503 descriptions), so they must not share a file.
_QUERY_TAG: str = "desc" if QUERY_MODE == "description" else "obj"

TOP_K_RETRIEVE: int = 25   # candidate pool from LanceDB (topk mode only); no config.yaml analog
TOP_K_FINAL:    int = _QCFG.get("top_k_final", 10)   # results after re-ranking (all fused into one 3D box)

# Retrieval A/B toggle (overridable via env for side-by-side runs):
#   "dynamic" — Otsu-sized pool + viewpoint-diverse subset (new pipeline)
#   "topk"    — legacy fixed-k retrieval, no diversity selection
RETRIEVAL_MODE: str = os.environ.get("EVAL_RETRIEVAL_MODE", _QCFG.get("retrieval_mode", "dynamic"))
N_DIVERSE: int = int(os.environ.get("EVAL_N_DIVERSE", _QCFG.get("n_diverse", 10)))

# 3D projection modes to score, cheapest first. Every mode reuses the same 2D
# results, so adding one costs a re-projection, not a re-detection. The other
# projection parameters (voxel, DBSCAN eps, min_instance_size, min_views, …)
# come from config.yaml's `projection` section — cluster_instances in
# particular is only meaningful with the min_views / min_instance_size gates
# configured there.
PROJECTION_MODES: tuple[str, ...] = tuple(
    m.strip()
    for m in os.environ.get(
        "EVAL_PROJECTION_MODES", "simple,cluster_single,cluster_instances"
    ).split(",")
    if m.strip()
)

# Per-run overrides of the config.yaml `projection` values, so reproducing an
# older run doesn't mean editing (and remembering to revert) config.yaml. Left
# unset, each falls back to the config. Whatever wins is what gets recorded in
# the results file's `projection_params`.
_PROJECTION_ENV: dict[str, type] = {
    "min_views": int,
    "min_instance_size": int,
    "voxel": float,
    "cluster_eps": float,
    "cluster_min_samples": int,
}

IOU_THRESHOLDS: tuple[float, ...] = (0.25, 0.50)

# Directory for the per-item results JSON (set to None to skip). One file per
# projection mode, suffixed with embedder + query mode + retrieval + projection
# mode so runs never overwrite each other.
RESULTS_DIR: Path | None = _ROOT

# ════════════════════════════════════════════════════════════════════════════
# END CONFIG
# ════════════════════════════════════════════════════════════════════════════

#: Column labels for the per-item console row.
_SHORT = {
    "simple": "simple",
    "cluster_single": "cl_single",
    "cluster_instances": "cl_inst",
}


def _effective_projection(cfg: dict) -> dict:
    """The ``projection`` config with any ``EVAL_<PARAM>`` env overrides applied.

    One dict serves both the pipeline and the results file, so what a run
    reports is always what it actually used.
    """
    pcfg = dict(cfg.get("projection") or {})
    for key, cast in _PROJECTION_ENV.items():
        raw = os.environ.get(f"EVAL_{key.upper()}")
        if raw is not None:
            pcfg[key] = cast(raw)
    return pcfg


def _check_modes() -> None:
    """Fail before models load if PROJECTION_MODES holds an unknown mode."""
    unknown = [m for m in PROJECTION_MODES if m not in _SHORT]
    if unknown or not PROJECTION_MODES:
        raise SystemExit(
            f"EVAL_PROJECTION_MODES: unknown mode(s) {unknown} — "
            f"expected a comma-separated subset of {', '.join(_SHORT)}."
        )


def _index_scene(scene_id: str, db, indexer) -> None:
    """Index *scene_id* into LanceDB if not already present."""
    from src.utils.datasets import load_scannet

    collection = f"scannet_{scene_id}"
    if collection in db.list_tables().tables:
        n = db.open_table(collection).count_rows()
        print(f"  [{scene_id}] already indexed ({n:,} rows) — skipping.")
        return

    scene_dir = SCANS_ROOT / scene_id
    if not scene_dir.exists():
        warnings.warn(f"Scene directory not found, skipping indexing: {scene_dir}")
        return

    print(f"  [{scene_id}] indexing …", flush=True)
    fs = load_scannet(scene_dir)
    indexer.insert(
        fs.paths,
        collection,
        depth_paths=fs.depth_paths,
        poses=fs.poses,
        intrinsics=fs.intrinsics,
        depth_scale=fs.depth_scale,
    )
    print(f"  [{scene_id}] indexed {len(fs):,} frames.")


def _build_pipeline(embedder, sam, db, cfg: dict, pcfg: dict):
    """Wire a Search2D from *cfg*, with the eval-specific CONFIG overrides applied.

    Everything the projection depends on (voxel, DBSCAN eps/min_samples,
    min_instance_size, min_views, …) comes from *pcfg* — the config's
    ``projection`` section after :func:`_effective_projection`; ``mode`` is set
    per query instead, so one pipeline serves all of PROJECTION_MODES.
    """
    from src.query.pipeline import Search2D

    qcfg = cfg.get("query") or {}
    return Search2D(
        embedder=embedder,
        detector=sam,
        db=db,
        retrieval_mode=RETRIEVAL_MODE,
        top_k_final=TOP_K_FINAL,
        min_k=qcfg.get("min_k", 10),
        max_k=qcfg.get("max_k", 400),
        min_separability=qcfg.get("min_separability", 0.75),
        strategy=qcfg.get("strategy", "tail"),
        n_diverse=N_DIVERSE,
        patch_frac=qcfg.get("patch_frac", 0.2),
        max_instances=pcfg.get("max_instances"),
        min_instance_size=pcfg.get("min_instance_size", 50),
        min_views=pcfg.get("min_views", 1),
        voxel=pcfg.get("voxel", 0.02),
        cluster_eps=pcfg.get("cluster_eps", 0.05),
        cluster_min_samples=pcfg.get("cluster_min_samples", 10),
        bbox_percentile=tuple(pcfg.get("bbox_percentile", (2.0, 98.0))),
    )


def main() -> None:
    from src.data_model import SearchState
    from src.eval import eval_items, gt_box, iou_aabb, load_val, val_scene_order
    from src.index import Indexer
    from src.models import SAMModel, db_path_for, embedder_name, load_embedder
    from src.utils.db import connect as db_connect

    _check_modes()

    cfg = _CFG
    embedder_id = embedder_name(cfg)
    db_path = db_path_for(cfg)
    pcfg = _effective_projection(cfg)

    # ── resolve scene subset ─────────────────────────────────────────────────
    scenes = SCENES if SCENES is not None else val_scene_order(SCANREFER_DIR)[:N_SCENES]
    print(f"Scenes      : {len(scenes)} ({', '.join(scenes[:3])}{'…' if len(scenes) > 3 else ''})")
    print(f"Embedder    : {embedder_id} "
          f"({(cfg['models'][embedder_id] or {}).get('model_id')})  →  {db_path}")
    print(f"Query mode  : {QUERY_MODE}")
    print(f"Retrieval   : {RETRIEVAL_MODE}"
          + (f" (n_diverse={N_DIVERSE})" if RETRIEVAL_MODE == "dynamic" else f" (top_k={TOP_K_RETRIEVE})"))
    print(f"Projection  : {', '.join(PROJECTION_MODES)}")
    print("Proj params : " + "  ".join(f"{k}={pcfg.get(k)}" for k in _PROJECTION_ENV))

    # ── load models once ─────────────────────────────────────────────────────
    print("\n─── Loading models ─────────────────────────────────────────────────")
    embedder = load_embedder(CONFIG_YAML)
    print(f"Embedder ready ({embedder_id}, dim={embedder.embedding_dim}).")
    sam = SAMModel.from_config(CONFIG_YAML)
    print("SAM ready.")

    # ── index any missing scenes ─────────────────────────────────────────────
    # Same store for indexing and querying, chosen by the embedder: a switch of
    # models.embedder means these scenes get indexed fresh rather than silently
    # searched with the wrong vectors.
    db = db_connect(db_path)
    indexer = Indexer(
        model=embedder,
        db_path=db_path,
        batch_size=cfg["indexing"].get("batch_size"),
    )
    print("\n─── Indexing ───────────────────────────────────────────────────────")
    for scene_id in scenes:
        _index_scene(scene_id, db, indexer)

    # ── build pipeline with the same model instances ─────────────────────────
    print("\n─── Building pipeline ──────────────────────────────────────────────")
    pipeline = _build_pipeline(embedder, sam, db, cfg, pcfg)
    # The 2D half is mode-independent: run it once per item, then re-project it
    # under each mode (see module docstring).
    chain_2d = (
        pipeline.embed | pipeline.retrieve | pipeline.select
        | pipeline.detect | pipeline.rerank
    )
    print("Pipeline ready (embed | retrieve | select | detect | rerank || project ×"
          f"{len(PROJECTION_MODES)}).")

    # ── build eval items ─────────────────────────────────────────────────────
    val = load_val(SCANREFER_DIR)
    items = eval_items(val, scenes, SCANS_ROOT, query_mode=QUERY_MODE)
    print(f"\n─── Evaluating {len(items)} items ──────────────────────────────────────")

    if not items:
        print("No eval items found — check SCENES / QUERY_MODE / SCANS_ROOT.")
        return

    # Cache GT boxes: one PLY/segs/aggregation read per (scene, object_id).
    gt_cache: dict[tuple[str, int], object] = {}

    records: dict[str, list[dict]] = {m: [] for m in PROJECTION_MODES}
    hits: dict[str, dict[float, int]] = {
        m: {thr: 0 for thr in IOU_THRESHOLDS} for m in PROJECTION_MODES
    }
    oracle_hits: dict[str, dict[float, int]] = {
        m: {thr: 0 for thr in IOU_THRESHOLDS} for m in PROJECTION_MODES
    }

    for idx_item, item in enumerate(items, 1):
        collection = f"scannet_{item.scene_id}"
        scene_dir  = SCANS_ROOT / item.scene_id

        # Ground-truth box (cached).
        gt_key = (item.scene_id, item.object_id)
        if gt_key not in gt_cache:
            try:
                gt_cache[gt_key] = gt_box(scene_dir, item.object_id)
            except Exception as exc:
                warnings.warn(f"gt_box failed for {gt_key}: {exc}")
                gt_cache[gt_key] = None
        box_gt = gt_cache[gt_key]

        # ── 2D half: shared by every projection mode ─────────────────────────
        state_2d = None
        diag = None
        t0 = time.perf_counter()
        try:
            state_2d = chain_2d.invoke(SearchState(
                query=item.query_text,
                collection_id=collection,
                top_k_retrieve=TOP_K_RETRIEVE,
                top_k_final=TOP_K_FINAL,
                retrieval_mode=RETRIEVAL_MODE,
                n_diverse=N_DIVERSE,
            ))
            diag = state_2d.retrieval_diag
        except Exception as exc:
            warnings.warn(f"2D pipeline failed for {gt_key}: {exc}")
        time_2d = time.perf_counter() - t0

        # ── 3D half: one projection per mode, same 2D results ────────────────
        cells = []
        for mode in PROJECTION_MODES:
            objects = []
            t1 = time.perf_counter()
            if state_2d is not None:
                try:
                    projected = pipeline.project.invoke(
                        state_2d.model_copy(update={"mode": mode})
                    ).projected
                    objects = projected or []
                except Exception as exc:
                    warnings.warn(f"projection {mode!r} failed for {gt_key}: {exc}")
            time_proj = time.perf_counter() - t1

            # Top-1 is the prediction (projected[0] = most multi-frame consensus
            # in cluster_instances, the only object otherwise). The oracle takes
            # the best of all returned instances: the gap between them is a
            # ranking failure, not a detection failure.
            ious = [iou_aabb(o.bbox, box_gt) for o in objects]
            iou = ious[0] if ious else 0.0
            oracle_iou = max(ious) if ious else 0.0
            oracle_rank = int(max(range(len(ious)), key=ious.__getitem__)) if ious else None

            for thr in IOU_THRESHOLDS:
                if iou >= thr:
                    hits[mode][thr] += 1
                if oracle_iou >= thr:
                    oracle_hits[mode][thr] += 1

            records[mode].append({
                "scene_id":    item.scene_id,
                "object_id":   item.object_id,
                "object_name": item.object_name,
                "ann_id":      item.ann_id,
                "query_text":  item.query_text,
                "iou":         round(iou, 4),
                "oracle_iou":  round(oracle_iou, 4),
                "oracle_rank": oracle_rank,
                "n_instances": len(objects),
                "query_time_s": round(time_2d + time_proj, 3),
                "project_time_s": round(time_proj, 3),
                **{f"hit@{thr:.2f}": bool(iou >= thr) for thr in IOU_THRESHOLDS},
                **{f"oracle_hit@{thr:.2f}": bool(oracle_iou >= thr) for thr in IOU_THRESHOLDS},
                "pool_size":    diag.pool_size if diag else None,
                "otsu_threshold": round(diag.threshold, 4) if diag and diag.threshold is not None else None,
                "separability": round(diag.separability, 4) if diag and diag.separability is not None else None,
                "gated":        diag.gated if diag else None,
                "n_selected":   diag.n_selected if diag else None,
            })

            flags = "".join("✓" if iou >= thr else "✗" for thr in IOU_THRESHOLDS)
            cell = f"{_SHORT[mode]} {iou:.3f}[{flags}]"
            if len(objects) > 1:
                cell += f" (n={len(objects)}, best {oracle_iou:.3f}@{oracle_rank})"
            cells.append(cell)

        # Per-item console row: one cell per mode.
        print(
            f"[{idx_item:4d}/{len(items)}] {item.scene_id}  obj={item.object_id:<4}"
            f"  {item.object_name:<20}  " + " | ".join(cells)
            + f"  t={time_2d:.2f}s  q={item.query_text[:40]!r}"
        )

    # ── summary ──────────────────────────────────────────────────────────────
    n = len(items)
    print("\n─── Results ────────────────────────────────────────────────────────")
    print(f"Embedder: {embedder_id}   Query mode: {QUERY_MODE}   "
          f"Retrieval: {RETRIEVAL_MODE}   N items: {n}")
    head = f"{'mode':<18}" + "".join(f"  Acc@{t:.2f}" for t in IOU_THRESHOLDS) \
           + "".join(f"  Orc@{t:.2f}" for t in IOU_THRESHOLDS) \
           + f"  {'inst':>5}  {'empty':>5}  {'s/query':>8}"
    print(head)
    print("─" * len(head))
    for mode in PROJECTION_MODES:
        recs = records[mode]
        total_time = sum(r["query_time_s"] for r in recs)
        n_inst = [r["n_instances"] for r in recs]
        print(
            f"{mode:<18}"
            + "".join(f"  {hits[mode][t] / n if n else 0.0:8.3f}" for t in IOU_THRESHOLDS)
            + "".join(f"  {oracle_hits[mode][t] / n if n else 0.0:8.3f}" for t in IOU_THRESHOLDS)
            + f"  {sum(n_inst) / n if n else 0.0:5.1f}"
            + f"  {sum(1 for c in n_inst if c == 0):5d}"
            + f"  {total_time / n if n else 0.0:8.2f}"
        )
    print("\nAcc = top-1 (projected[0]) · Orc = oracle best-of-instances "
          "· inst = mean #objects returned · empty = items with no 3D object")
    print("s/query = shared 2D time + that mode's projection — the modes share the "
          "2D half, so the wall-clock\n          total is not their sum (see "
          "project_time_s per item for the projection alone).")

    if RESULTS_DIR is None:
        return

    for mode in PROJECTION_MODES:
        recs = records[mode]
        total_time = sum(r["query_time_s"] for r in recs)
        pool_sizes = [r["pool_size"] for r in recs if r["pool_size"] is not None]
        n_inst = [r["n_instances"] for r in recs]
        payload = {
            "embedder": embedder_id,
            "embedder_model_id": (cfg["models"][embedder_id] or {}).get("model_id"),
            "query_mode": QUERY_MODE,
            "retrieval_mode": RETRIEVAL_MODE,
            "projection_mode": mode,
            "projection_params": pcfg,
            "n_diverse": N_DIVERSE if RETRIEVAL_MODE == "dynamic" else None,
            "scenes": scenes,
            "n_items": len(recs),
            "summary": {
                **{f"acc@{thr:.2f}": hits[mode][thr] / n if n else 0.0 for thr in IOU_THRESHOLDS},
                **{f"oracle_acc@{thr:.2f}": oracle_hits[mode][thr] / n if n else 0.0
                   for thr in IOU_THRESHOLDS},
                "query_time_total_s": round(total_time, 3),
                "query_time_avg_s": round(total_time / n, 3) if n else 0.0,
                "gated_count": sum(1 for r in recs if r["gated"]),
                "pool_size_mean": round(sum(pool_sizes) / len(pool_sizes), 1) if pool_sizes else None,
                "n_instances_mean": round(sum(n_inst) / n, 2) if n else None,
                "empty_count": sum(1 for c in n_inst if c == 0),
            },
            "items": recs,
        }
        out = (RESULTS_DIR /
               f"eval_results_scanrefer_{embedder_id}_{_QUERY_TAG}_"
               f"{RETRIEVAL_MODE}_{mode}.json")
        out.write_text(json.dumps(payload, indent=2))
        print(f"Results written to: {out}")


if __name__ == "__main__":
    main()
