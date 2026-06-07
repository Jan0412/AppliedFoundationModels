"""Tests for src.index.indexer.Indexer.

The Indexer is tested against a real LanceDB on a temp directory (LanceDB is
file-based, no server needed) so we also catch any drift in the LanceDB API.
SigLIP is mocked via conftest.mock_siglip_model.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from src.index import Indexer, JobRegistry, get_status
from src.utils.db import load_collection_meta


# ---------------------------------------------------------------------------
# Calibration helpers
#
# insert/update now require per-frame depth_paths + poses and per-collection
# intrinsics + depth_scale. These factories build sensible defaults so each
# test only spells out the calibration it actually asserts on.
# ---------------------------------------------------------------------------


_INTRINSICS = {"fx": 100.0, "fy": 100.0, "cx": 16.0, "cy": 16.0}


def _poses(n: int) -> list[np.ndarray]:
    return [np.eye(4, dtype=np.float32) for _ in range(n)]


def _norm_calib(n: int, depth_paths: list[str] | None = None) -> dict:
    """Per-call args for ``_normalize`` (depth_paths + poses only)."""
    return {
        "depth_paths": depth_paths
        if depth_paths is not None
        else [f"/depth/{i:04d}.png" for i in range(n)],
        "poses": _poses(n),
    }


def _calib(n: int, depth_paths: list[str] | None = None) -> dict:
    """Full per-call args for ``insert``/``update`` (adds collection calibration)."""
    return {
        **_norm_calib(n, depth_paths),
        "intrinsics": dict(_INTRINSICS),
        "depth_scale": 1000.0,
    }


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_uses_model_batch_size_when_not_overridden(mock_siglip_model, tmp_db_path):
    idx = Indexer(model=mock_siglip_model, db_path=tmp_db_path)
    assert idx.batch_size == mock_siglip_model.batch_size


def test_init_overrides_batch_size(mock_siglip_model, tmp_db_path):
    idx = Indexer(model=mock_siglip_model, db_path=tmp_db_path, batch_size=3)
    assert idx.batch_size == 3


def test_init_accepts_pathlib_path(mock_siglip_model, tmp_db_path):
    idx = Indexer(model=mock_siglip_model, db_path=Path(tmp_db_path))
    assert idx.db is not None


def test_init_stores_model_reference(mock_siglip_model, tmp_db_path):
    idx = Indexer(model=mock_siglip_model, db_path=tmp_db_path)
    assert idx.model is mock_siglip_model


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_reads_indexing_section(tmp_indexing_config, mock_siglip_model):
    """from_config reads `indexing.db_path` and `indexing.batch_size`."""
    with patch(
        "src.index.indexer.SigLIPModel.from_config",
        return_value=mock_siglip_model,
    ):
        idx = Indexer.from_config(tmp_indexing_config)
    assert idx.batch_size == 4                     # from indexing.batch_size
    assert idx.model is mock_siglip_model


def test_from_config_falls_back_to_model_batch_size_when_unset(tmp_path, mock_siglip_model):
    """When indexing.batch_size is absent, the model's batch_size is used."""
    import yaml as _yaml
    cfg = {
        "models": {
            "siglip": {"model_id": "x", "device": "cpu", "batch_size": 99},
            "sam": {"model_id": "y", "device": "cpu", "threshold": 0.5, "mask_threshold": 0.5},
        },
        "indexing": {"db_path": str(tmp_path / "db")},
    }
    p = tmp_path / "config.yaml"
    p.write_text(_yaml.dump(cfg))

    with patch(
        "src.index.indexer.SigLIPModel.from_config",
        return_value=mock_siglip_model,
    ):
        idx = Indexer.from_config(p)
    assert idx.batch_size == mock_siglip_model.batch_size


# ---------------------------------------------------------------------------
# _schema
# ---------------------------------------------------------------------------


def test_schema_has_expected_fields(indexer):
    schema = indexer._schema()
    names = [f.name for f in schema]
    assert names == ["id", "collection_id", "vector", "path", "depth_path", "cam2world"]


def test_schema_vector_dim_matches_model(indexer, mock_siglip_model):
    """The fixed-size list dimension equals model.embedding_dim."""
    schema = indexer._schema()
    vec_field = schema.field("vector")
    assert isinstance(vec_field.type, pa.FixedSizeListType)
    assert vec_field.type.list_size == mock_siglip_model.embedding_dim


def test_schema_cam2world_is_float_list(indexer):
    # Variable-length list (not fixed-size) so LanceDB doesn't treat it as a
    # second vector column; the 16-element invariant lives in _flatten_pose.
    schema = indexer._schema()
    cam_field = schema.field("cam2world")
    assert pa.types.is_list(cam_field.type)
    assert not isinstance(cam_field.type, pa.FixedSizeListType)


def test_schema_id_and_collection_are_strings(indexer):
    schema = indexer._schema()
    assert schema.field("id").type == pa.string()
    assert schema.field("collection_id").type == pa.string()


def test_meta_schema_has_expected_fields(indexer):
    names = [f.name for f in indexer._meta_schema()]
    assert names == ["collection_id", "fx", "fy", "cx", "cy", "depth_scale"]


# ---------------------------------------------------------------------------
# _open_or_create_table
# ---------------------------------------------------------------------------


def test_open_or_create_table_creates_new_when_missing(indexer):
    table = indexer._open_or_create_table("newcoll")
    assert table.count_rows() == 0
    assert "newcoll" in indexer.db.list_tables().tables


def test_open_or_create_table_returns_existing(indexer):
    t1 = indexer._open_or_create_table("dup")
    t1.add([{
        "id": "row1", "collection_id": "dup",
        "vector": [0.0] * 8, "path": "",
        "depth_path": "", "cam2world": [0.0] * 16,
    }])
    t2 = indexer._open_or_create_table("dup")
    assert t2.count_rows() == 1


# ---------------------------------------------------------------------------
# _normalize — id derivation
# ---------------------------------------------------------------------------


def test_normalize_paths_derive_sha1_ids(indexer):
    records = indexer._normalize(
        ["/foo.png", "/bar.png"], collection_id="c", ids=None, **_norm_calib(2)
    )
    expected_foo = hashlib.sha1(b"c:/foo.png").hexdigest()
    expected_bar = hashlib.sha1(b"c:/bar.png").hexdigest()
    assert records[0]["id"] == expected_foo
    assert records[1]["id"] == expected_bar


def test_normalize_paths_ids_differ_per_collection(indexer):
    r1 = indexer._normalize(["/x.png"], collection_id="A", ids=None, **_norm_calib(1))
    r2 = indexer._normalize(["/x.png"], collection_id="B", ids=None, **_norm_calib(1))
    assert r1[0]["id"] != r2[0]["id"]


def test_normalize_paths_ids_are_stable(indexer):
    """Same input → same id on repeat calls."""
    r1 = indexer._normalize(["/p.png"], collection_id="c", ids=None, **_norm_calib(1))
    r2 = indexer._normalize(["/p.png"], collection_id="c", ids=None, **_norm_calib(1))
    assert r1[0]["id"] == r2[0]["id"]


def test_normalize_explicit_ids_override_derivation(indexer):
    records = indexer._normalize(
        ["/a.png", "/b.png"],
        collection_id="c",
        ids=["custom1", "custom2"],
        **_norm_calib(2),
    )
    assert [r["id"] for r in records] == ["custom1", "custom2"]


def test_normalize_pil_derives_id_from_png_bytes(indexer, fake_pil_image):
    records = indexer._normalize(
        [fake_pil_image], collection_id="c", ids=None, **_norm_calib(1)
    )
    buf = io.BytesIO()
    fake_pil_image.save(buf, format="PNG")
    expected = hashlib.sha1(buf.getvalue()).hexdigest()
    assert records[0]["id"] == expected


def test_normalize_pil_explicit_id_overrides(indexer, fake_pil_image):
    records = indexer._normalize(
        [fake_pil_image], collection_id="c", ids=["mine"], **_norm_calib(1)
    )
    assert records[0]["id"] == "mine"


def test_normalize_accepts_pathlib_path(indexer):
    records = indexer._normalize(
        [Path("/some/p.png")], collection_id="c", ids=None, **_norm_calib(1)
    )
    assert records[0]["path"] == "/some/p.png"


# ---------------------------------------------------------------------------
# _normalize — depth_path + cam2world
# ---------------------------------------------------------------------------


def test_normalize_stores_depth_path(indexer):
    r = indexer._normalize(
        ["/a.png", "/b.png"], collection_id="c", ids=None,
        depth_paths=["/d0.png", "/d1.png"], poses=_poses(2),
    )
    assert r[0]["depth_path"] == "/d0.png"
    assert r[1]["depth_path"] == "/d1.png"


def test_normalize_flattens_pose_row_major(indexer):
    pose = np.arange(16, dtype=np.float32).reshape(4, 4)
    r = indexer._normalize(
        ["/a.png"], collection_id="c", ids=None,
        depth_paths=["/d.png"], poses=[pose],
    )
    assert r[0]["cam2world"] == [float(i) for i in range(16)]


def test_normalize_bad_pose_shape_raises(indexer):
    with pytest.raises(ValueError, match="4x4"):
        indexer._normalize(
            ["/a.png"], collection_id="c", ids=None,
            depth_paths=["/d.png"], poses=[np.eye(3)],
        )


# ---------------------------------------------------------------------------
# _normalize — lambda load callables (no late-binding bugs)
# ---------------------------------------------------------------------------


def test_normalize_load_callable_loads_pil_from_path(indexer, make_image_files):
    paths = make_image_files(2)
    records = indexer._normalize(paths, collection_id="c", ids=None, **_norm_calib(2))
    img0 = records[0]["load"]()
    img1 = records[1]["load"]()
    assert isinstance(img0, Image.Image)
    assert isinstance(img1, Image.Image)
    # Lambda-capture sanity: each record loads its OWN path, not the last one
    assert img0 != img1   # different content (different colours)


def test_normalize_load_callable_returns_rgb_image(indexer, make_image_files):
    paths = make_image_files(1)
    records = indexer._normalize(paths, collection_id="c", ids=None, **_norm_calib(1))
    assert records[0]["load"]().mode == "RGB"


def test_normalize_load_callable_pil_unchanged_when_rgb(indexer, fake_pil_image):
    records = indexer._normalize(
        [fake_pil_image], collection_id="c", ids=None, **_norm_calib(1)
    )
    assert records[0]["load"]() is fake_pil_image


def test_normalize_load_callable_pil_converts_non_rgb(indexer):
    grey = Image.new("L", (8, 8), color=128)
    records = indexer._normalize([grey], collection_id="c", ids=None, **_norm_calib(1))
    loaded = records[0]["load"]()
    assert loaded.mode == "RGB"


# ---------------------------------------------------------------------------
# _normalize — error cases
# ---------------------------------------------------------------------------


def test_normalize_mismatched_ids_length_raises(indexer):
    with pytest.raises(ValueError, match="len\\(ids\\)"):
        indexer._normalize(
            ["/a.png", "/b.png"], collection_id="c", ids=["only-one"], **_norm_calib(2)
        )


def test_normalize_missing_depth_paths_raises(indexer):
    with pytest.raises(ValueError, match="depth_paths"):
        indexer._normalize(
            ["/a.png"], collection_id="c", ids=None, depth_paths=None, poses=_poses(1)
        )


def test_normalize_mismatched_depth_paths_length_raises(indexer):
    with pytest.raises(ValueError, match="depth_paths"):
        indexer._normalize(
            ["/a.png", "/b.png"], collection_id="c", ids=None,
            depth_paths=["/d.png"], poses=_poses(2),
        )


def test_normalize_missing_poses_raises(indexer):
    with pytest.raises(ValueError, match="poses"):
        indexer._normalize(
            ["/a.png"], collection_id="c", ids=None, depth_paths=["/d.png"], poses=None
        )


def test_normalize_mismatched_poses_length_raises(indexer):
    with pytest.raises(ValueError, match="poses"):
        indexer._normalize(
            ["/a.png", "/b.png"], collection_id="c", ids=None,
            depth_paths=["/d0.png", "/d1.png"], poses=_poses(1),
        )


def test_normalize_duplicate_ids_within_call_raises(indexer):
    with pytest.raises(ValueError, match="[Dd]uplicate ids"):
        indexer._normalize(
            ["/a.png", "/b.png"],
            collection_id="c",
            ids=["same", "same"],
            **_norm_calib(2),
        )


def test_normalize_unsupported_image_type_raises_typeerror(indexer):
    with pytest.raises(TypeError, match="must be"):
        indexer._normalize([42], collection_id="c", ids=None, **_norm_calib(1))


# ---------------------------------------------------------------------------
# insert — happy path
# ---------------------------------------------------------------------------


def test_insert_writes_one_row_per_image(indexer, make_image_files):
    paths = make_image_files(5)
    job = indexer.insert(paths, collection_id="c1", **_calib(5))
    table = indexer.db.open_table("c1")
    assert table.count_rows() == 5
    assert job.processed == 5
    assert job.total == 5
    assert job.state == "done"


def test_insert_returns_jobstatus(indexer, make_image_files):
    paths = make_image_files(3)
    job = indexer.insert(paths, collection_id="c2", **_calib(3))
    assert job.collection_id == "c2"
    assert job.state == "done"
    assert job.finished_at is not None


def test_insert_uses_supplied_job_id(indexer, make_image_files):
    paths = make_image_files(2)
    job = indexer.insert(paths, collection_id="c3", job_id="my-job-77", **_calib(2))
    assert job.job_id == "my-job-77"
    assert get_status("my-job-77") is not None


def test_insert_invokes_model_embed_images(indexer, mock_siglip_model, make_image_files):
    paths = make_image_files(3)
    indexer.insert(paths, collection_id="c4", **_calib(3))
    assert mock_siglip_model.embed_images.called
    # Each call receives a list of PIL.Image
    for call in mock_siglip_model.embed_images.call_args_list:
        (pil_list,), _ = call
        assert all(isinstance(im, Image.Image) for im in pil_list)


def test_insert_batches_inputs_into_batch_size_chunks(indexer, mock_siglip_model, make_image_files):
    """batch_size=2 and 5 images → 3 embed_images calls (2 + 2 + 1)."""
    paths = make_image_files(5)
    indexer.insert(paths, collection_id="c5", **_calib(5))
    sizes = [len(c.args[0]) for c in mock_siglip_model.embed_images.call_args_list]
    assert sizes == [2, 2, 1]


def test_insert_stores_collection_id_on_every_row(indexer, make_image_files):
    paths = make_image_files(3)
    indexer.insert(paths, collection_id="myroom", **_calib(3))
    rows = indexer.db.open_table("myroom").to_arrow().to_pylist()
    assert all(r["collection_id"] == "myroom" for r in rows)


def test_insert_stores_path_depth_and_pose(indexer, make_image_files):
    paths = make_image_files(2)
    pose0 = np.arange(16, dtype=np.float32).reshape(4, 4)
    pose1 = np.eye(4, dtype=np.float32)
    indexer.insert(
        paths, collection_id="c6",
        depth_paths=["/d0.png", "/d1.png"], poses=[pose0, pose1],
        intrinsics=dict(_INTRINSICS), depth_scale=1000.0,
    )
    rows_by_path = {r["path"]: r for r in indexer.db.open_table("c6").to_arrow().to_pylist()}
    assert rows_by_path[paths[0]]["depth_path"] == "/d0.png"
    assert rows_by_path[paths[1]]["depth_path"] == "/d1.png"
    assert rows_by_path[paths[0]]["cam2world"] == [float(i) for i in range(16)]


def test_insert_writes_collection_meta(indexer, make_image_files):
    paths = make_image_files(2)
    indexer.insert(
        paths, collection_id="cm", **_norm_calib(2),
        intrinsics={"fx": 5.0, "fy": 6.0, "cx": 7.0, "cy": 8.0}, depth_scale=1234.0,
    )
    meta = load_collection_meta(indexer.db, "cm")
    assert meta == {"fx": 5.0, "fy": 6.0, "cx": 7.0, "cy": 8.0, "depth_scale": 1234.0}


def test_insert_missing_calibration_raises(indexer, make_image_files):
    paths = make_image_files(2)
    with pytest.raises(TypeError):
        # depth_paths/poses/intrinsics/depth_scale are required keyword args
        indexer.insert(paths, collection_id="cmiss")


def test_insert_writes_vector_of_embedding_dim(indexer, mock_siglip_model, make_image_files):
    paths = make_image_files(2)
    indexer.insert(paths, collection_id="c7", **_calib(2))
    rows = indexer.db.open_table("c7").to_arrow().to_pylist()
    for r in rows:
        assert len(r["vector"]) == mock_siglip_model.embedding_dim


def test_insert_accepts_pil_images(indexer, fake_pil_image):
    imgs = [fake_pil_image, Image.new("RGB", (16, 16), color=(0, 0, 0))]
    job = indexer.insert(imgs, collection_id="c8", ids=["a", "b"], **_calib(2))
    assert job.state == "done"
    rows = indexer.db.open_table("c8").to_arrow().to_pylist()
    assert {r["id"] for r in rows} == {"a", "b"}


def test_insert_advances_processed_to_total(indexer, make_image_files):
    paths = make_image_files(7)
    job = indexer.insert(paths, collection_id="c9", **_calib(7))
    assert job.processed == job.total == 7


# ---------------------------------------------------------------------------
# insert — conflict detection
# ---------------------------------------------------------------------------


def test_insert_raises_when_id_already_exists(indexer, make_image_files):
    paths = make_image_files(3)
    indexer.insert(paths, collection_id="c10", **_calib(3))
    with pytest.raises(ValueError, match="already exist"):
        indexer.insert(paths, collection_id="c10", **_calib(3))


def test_insert_failure_marks_job_failed(indexer, make_image_files):
    paths = make_image_files(2)
    indexer.insert(paths, collection_id="c11", **_calib(2))
    with pytest.raises(ValueError):
        indexer.insert(paths, collection_id="c11", job_id="dead-job", **_calib(2))
    # The job for the failed insert is registered with state='failed'.
    # (The first insert succeeded, the second one — under job id "dead-job" —
    # never reaches embed because conflict-check runs first; therefore no job
    # is started in that case.)  Verify by re-raising on a different code path:
    # an embed-time failure.
    indexer.model.embed_images.side_effect = RuntimeError("explode")
    paths2 = make_image_files(2, subdir="more")
    with pytest.raises(RuntimeError, match="explode"):
        indexer.insert(paths2, collection_id="c11b", job_id="boom", **_calib(2))
    s = get_status("boom")
    assert s is not None
    assert s["state"] == "failed"
    assert "explode" in (s["error"] or "")


def test_insert_into_empty_table_skips_conflict_check(indexer, make_image_files):
    """No existing rows → no fetch of existing ids → no error."""
    paths = make_image_files(2)
    job = indexer.insert(paths, collection_id="fresh", **_calib(2))
    assert job.state == "done"


# ---------------------------------------------------------------------------
# update — happy path
# ---------------------------------------------------------------------------


def test_update_on_empty_table_inserts(indexer, make_image_files):
    paths = make_image_files(3)
    job = indexer.update(paths, collection_id="u1", **_calib(3))
    assert job.state == "done"
    assert indexer.db.open_table("u1").count_rows() == 3


def test_update_replaces_matching_rows(indexer, make_image_files):
    paths = make_image_files(3)
    indexer.insert(
        paths, collection_id="u2",
        **_calib(3, depth_paths=["/a.png", "/b.png", "/c.png"]),
    )
    # Re-embed the same paths with different depth paths; row count stays at 3,
    # but the depth_path columns reflect the new values.
    indexer.update(
        paths, collection_id="u2",
        **_calib(3, depth_paths=["/x.png", "/y.png", "/z.png"]),
    )
    table = indexer.db.open_table("u2")
    assert table.count_rows() == 3
    rows = {r["path"]: r["depth_path"] for r in table.to_arrow().to_pylist()}
    assert rows[paths[0]] == "/x.png"
    assert rows[paths[1]] == "/y.png"
    assert rows[paths[2]] == "/z.png"


def test_update_inserts_new_ids_alongside_existing(indexer, make_image_files):
    initial = make_image_files(2, subdir="a")
    indexer.insert(initial, collection_id="u3", **_calib(2))
    extra = make_image_files(2, subdir="b")
    indexer.update(extra, collection_id="u3", **_calib(2))
    assert indexer.db.open_table("u3").count_rows() == 4


def test_update_does_not_raise_on_duplicate_ids_across_calls(indexer, make_image_files):
    paths = make_image_files(2)
    indexer.insert(paths, collection_id="u4", **_calib(2))
    # Same paths → same ids; should upsert without complaining.
    job = indexer.update(paths, collection_id="u4", **_calib(2))
    assert job.state == "done"
    assert indexer.db.open_table("u4").count_rows() == 2


def test_update_failure_marks_job_failed_and_reraises(indexer, make_image_files):
    paths = make_image_files(2)
    indexer.model.embed_images.side_effect = RuntimeError("upd-fail")
    with pytest.raises(RuntimeError, match="upd-fail"):
        indexer.update(paths, collection_id="u5", job_id="updboom", **_calib(2))
    s = get_status("updboom")
    assert s is not None
    assert s["state"] == "failed"
    assert "upd-fail" in (s["error"] or "")


# ---------------------------------------------------------------------------
# End-to-end sanity: insert + LanceDB vector search
# ---------------------------------------------------------------------------


def test_insert_then_vector_search_returns_inserted_rows(indexer, make_image_files):
    paths = make_image_files(4)
    indexer.insert(paths, collection_id="e2e", **_calib(4))
    table = indexer.db.open_table("e2e")
    # Use the embedding of the first row as the query — top hit must be itself.
    rows = table.to_arrow().to_pylist()
    query_vec = rows[0]["vector"]
    hits = table.search(query_vec).metric("cosine").limit(1).to_list()
    assert hits[0]["id"] == rows[0]["id"]
    assert hits[0]["collection_id"] == "e2e"


# ---------------------------------------------------------------------------
# Status integration — JobRegistry holds the live job during the run
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    JobRegistry._jobs.clear()
    yield
    JobRegistry._jobs.clear()


def test_get_status_during_insert_reflects_progress(indexer, mock_siglip_model, make_image_files):
    """Verify that as embed_images runs, the registered job's `processed`
    counter visibly increases (snapshot taken from inside embed_images)."""
    paths = make_image_files(6)        # batch_size=2 → 3 batches
    snapshots: list[int] = []

    original_embed = mock_siglip_model.embed_images.side_effect

    def _spy(pil_images):
        # capture the registered job's processed count BEFORE this batch lands
        jobs = JobRegistry.list_all()
        if jobs:
            snapshots.append(jobs[0].processed)
        return original_embed(pil_images)

    mock_siglip_model.embed_images.side_effect = _spy

    indexer.insert(paths, collection_id="prog", **_calib(6))

    # Before batch 1: 0;  before batch 2: 2;  before batch 3: 4.
    assert snapshots == [0, 2, 4]
