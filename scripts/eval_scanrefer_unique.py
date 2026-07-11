#!/usr/bin/env python3
"""ScanRefer benchmark (unique-object subset) for the Search2D pipeline.

Runs the pipeline on ScanRefer val scenes, restricted to objects whose category
appears exactly once in the scene ("unique" objects). Reports Acc@0.25 / Acc@0.5.

Usage
-----
Edit the CONFIG block below, then:

    cd /path/to/AppliedFoundationModels
    uv run python scripts/eval_scanrefer_unique.py

Re-runs are cheap: scenes already in LanceDB are skipped at the indexing step.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

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


# Subset: either set SCENES to an explicit list of scene ids, or leave it None
# and set N_SCENES to use the first N scenes from ScanRefer_filtered_val.txt.
SCENES: list[str] | None = None
N_SCENES: int = 141

# Query mode:
#   "object_name"  — one query per unique (scene, object_id);  query = category noun.
#   "description"  — one query per val description of a unique object; query = sentence.
QUERY_MODE: str = "object_name"

TOP_K_RETRIEVE: int = 25   # candidate pool from LanceDB (topk mode only)
TOP_K_FINAL:    int = 15   # results after re-ranking (all fused into one 3D box)

# Retrieval A/B toggle (overridable via env for side-by-side runs):
#   "dynamic" — Otsu-sized pool + viewpoint-diverse subset (new pipeline)
#   "topk"    — legacy fixed-k retrieval, no diversity selection
RETRIEVAL_MODE: str = os.environ.get("EVAL_RETRIEVAL_MODE", "dynamic")
N_DIVERSE: int = int(os.environ.get("EVAL_N_DIVERSE", "25"))

IOU_THRESHOLDS: tuple[float, ...] = (0.25, 0.50)

# Where to write the per-item results JSON (set to None to skip).
# Suffixed with the retrieval mode so A/B runs don't overwrite each other.
RESULTS_OUT: Path | None = _ROOT / f"eval_results_scanrefer_{RETRIEVAL_MODE}.json"

# ════════════════════════════════════════════════════════════════════════════
# END CONFIG
# ════════════════════════════════════════════════════════════════════════════


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


def main() -> None:
    import yaml
    from src.eval import eval_items, gt_box, iou_aabb, load_val, val_scene_order
    from src.index import Indexer
    from src.models import SAMModel, SigLIPModel
    from src.query.pipeline import Search2D
    from src.utils.db import connect as db_connect

    # ── resolve scene subset ─────────────────────────────────────────────────
    scenes = SCENES if SCENES is not None else val_scene_order(SCANREFER_DIR)[:N_SCENES]
    print(f"Scenes      : {len(scenes)} ({', '.join(scenes[:3])}{'…' if len(scenes) > 3 else ''})")
    print(f"Query mode  : {QUERY_MODE}")
    print(f"Retrieval   : {RETRIEVAL_MODE}"
          + (f" (n_diverse={N_DIVERSE})" if RETRIEVAL_MODE == "dynamic" else f" (top_k={TOP_K_RETRIEVE})"))

    # ── load models once ─────────────────────────────────────────────────────
    print("\n─── Loading models ─────────────────────────────────────────────────")
    siglip = SigLIPModel.from_config(CONFIG_YAML)
    print("SigLIP ready.")
    sam = SAMModel.from_config(CONFIG_YAML)
    print("SAM ready.")

    # ── index any missing scenes ─────────────────────────────────────────────
    db = db_connect(str(_ROOT / "data" / "lancedb"))
    cfg = yaml.safe_load(CONFIG_YAML.read_text())
    indexer = Indexer(
        model=siglip,
        db_path=cfg["indexing"]["db_path"],
        batch_size=cfg["indexing"].get("batch_size"),
    )
    print("\n─── Indexing ───────────────────────────────────────────────────────")
    for scene_id in scenes:
        _index_scene(scene_id, db, indexer)

    # ── build pipeline with the same model instances ─────────────────────────
    print("\n─── Building pipeline ──────────────────────────────────────────────")
    pipeline = Search2D(siglip=siglip, detector=sam, db=db, n_diverse=N_DIVERSE)
    print("Pipeline ready (embed | retrieve | select | detect | rerank | project).")

    # ── build eval items ─────────────────────────────────────────────────────
    val = load_val(SCANREFER_DIR)
    items = eval_items(val, scenes, SCANS_ROOT, query_mode=QUERY_MODE)
    print(f"\n─── Evaluating {len(items)} items ──────────────────────────────────────")

    if not items:
        print("No eval items found — check SCENES / QUERY_MODE / SCANS_ROOT.")
        return

    # Cache GT boxes: one PLY/segs/aggregation read per (scene, object_id).
    gt_cache: dict[tuple[str, int], object] = {}

    records = []
    hits = {thr: 0 for thr in IOU_THRESHOLDS}

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

        # Pipeline prediction.
        pred_bbox = None
        diag = None
        t0 = time.perf_counter()
        try:
            state = pipeline.invoke(
                query=item.query_text,
                collection_id=collection,
                top_k_retrieve=TOP_K_RETRIEVE,
                top_k_final=TOP_K_FINAL,
                retrieval_mode=RETRIEVAL_MODE,
                n_diverse=N_DIVERSE,
            )
            diag = state.retrieval_diag
            if state.projected:
                pred_bbox = state.projected[0].bbox
        except Exception as exc:
            warnings.warn(f"pipeline failed for {gt_key}: {exc}")

        t1 = time.perf_counter()
        query_time = t1 - t0

        iou = iou_aabb(pred_bbox, box_gt)
        hit_flags = {thr: (iou >= thr) for thr in IOU_THRESHOLDS}
        for thr in IOU_THRESHOLDS:
            if hit_flags[thr]:
                hits[thr] += 1

        # Per-item console row.
        flags_str = "  ".join(
            f"@{thr:.2f}={'✓' if hit_flags[thr] else '✗'}"
            for thr in IOU_THRESHOLDS
        )
        print(
            f"[{idx_item:4d}/{len(items)}] {item.scene_id}  obj={item.object_id:<4}"
            f"  {item.object_name:<20}  IoU={iou:.3f}  {flags_str}"
            f"  t={query_time:.2f}s  q={item.query_text[:60]!r}"
        )

        records.append({
            "scene_id":    item.scene_id,
            "object_id":   item.object_id,
            "object_name": item.object_name,
            "ann_id":      item.ann_id,
            "query_text":  item.query_text,
            "iou":         round(iou, 4),
            "query_time_s": round(query_time, 3),
            **{f"hit@{thr:.2f}": bool(hit_flags[thr]) for thr in IOU_THRESHOLDS},
            "pool_size":    diag.pool_size if diag else None,
            "otsu_threshold": round(diag.threshold, 4) if diag and diag.threshold is not None else None,
            "separability": round(diag.separability, 4) if diag and diag.separability is not None else None,
            "gated":        diag.gated if diag else None,
            "n_selected":   diag.n_selected if diag else None,
        })

    # ── summary ──────────────────────────────────────────────────────────────
    n = len(records)
    total_time = sum(r["query_time_s"] for r in records)
    print("\n─── Results ────────────────────────────────────────────────────────")
    print(f"Query mode : {QUERY_MODE}")
    print(f"N items    : {n}")
    for thr in IOU_THRESHOLDS:
        acc = hits[thr] / n if n else 0.0
        print(f"Acc@{thr:.2f}   : {hits[thr]:3d}/{n}  = {acc:.3f}")
    print(f"Time total : {total_time:.1f}s  avg {total_time / n:.2f}s/query" if n else "Time total : —")

    if RESULTS_OUT is not None:
        pool_sizes = [r["pool_size"] for r in records if r["pool_size"] is not None]
        payload = {
            "query_mode": QUERY_MODE,
            "retrieval_mode": RETRIEVAL_MODE,
            "n_diverse": N_DIVERSE if RETRIEVAL_MODE == "dynamic" else None,
            "scenes": scenes,
            "n_items": n,
            "summary": {
                **{f"acc@{thr:.2f}": hits[thr] / n if n else 0.0 for thr in IOU_THRESHOLDS},
                "query_time_total_s": round(total_time, 3),
                "query_time_avg_s": round(total_time / n, 3) if n else 0.0,
                "gated_count": sum(1 for r in records if r["gated"]),
                "pool_size_mean": round(sum(pool_sizes) / len(pool_sizes), 1) if pool_sizes else None,
            },
            "items": records,
        }
        Path(RESULTS_OUT).write_text(json.dumps(payload, indent=2))
        print(f"\nResults written to: {RESULTS_OUT}")


if __name__ == "__main__":
    main()
