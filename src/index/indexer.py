"""Image indexer: SigLIP embeddings → LanceDB.

One LanceDB table per ``collection_id``.  Every row has an explicit ``id``
(caller-supplied or SHA-1 derived) and a ``collection_id`` column so that
multiple benchmarks / rooms can coexist in the same DB directory.

Example — first-time index of a TUM RGB-D benchmark::

    from pathlib import Path
    from src.index import Indexer, get_status

    idx = Indexer.from_config("config.yaml")

    paths = sorted(str(p) for p in Path("data/tum/fr1_desk/rgb").glob("*.png"))
    job   = idx.insert(paths, collection_id="fr1_desk")   # live tqdm bar

    print(get_status(job.job_id))
    # {'state': 'done', 'total': 573, 'processed': 573, ...}

Re-embed a subset and upsert::

    job2 = idx.update(new_paths, collection_id="fr1_desk")
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import yaml
from PIL import Image
from tqdm.auto import tqdm

from src.models import SigLIPModel
from src.utils.db import connect as _db_connect

from .status import JobRegistry, JobStatus


def _flatten_pose(pose) -> list[float]:
    """Coerce a 4x4 camera-to-world pose into a 16-float row-major list.

    Accepts a NumPy array or nested sequence shaped ``(4, 4)``. Raises
    :class:`ValueError` for any other shape so a malformed pose is caught at
    index time rather than producing silently wrong 3D points later.
    """
    arr = np.asarray(pose, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(
            f"each pose must be a 4x4 cam2world matrix; got shape {arr.shape}."
        )
    return arr.reshape(16).tolist()


class Indexer:
    """Embed images with SigLIP and write them to LanceDB.

    One LanceDB table per ``collection_id``; the table name *is* the
    collection id.  Rows are keyed by ``id``.

    Public methods:

    - :meth:`insert` — add new rows; raises :class:`ValueError` if any id
      already exists in the table.
    - :meth:`update` — upsert rows (replace on match, insert on miss).

    Both return a :class:`~src.index.status.JobStatus` whose ``as_dict()``
    is the serialisable payload the future API will serve.
    """

    def __init__(
        self,
        model: SigLIPModel,
        db_path: str | Path,
        batch_size: int | None = None,
    ) -> None:
        """
        Args:
            model:      Pre-loaded :class:`SigLIPModel`.  ``embed_images`` is
                        the only method called; it is reused verbatim.
            db_path:    Directory that LanceDB uses as its root store.
            batch_size: Images per embedding call.  Defaults to
                        ``model.batch_size`` when ``None``.
        """
        self.model = model
        self.db = _db_connect(db_path)
        self.batch_size = batch_size if batch_size is not None else model.batch_size

    # ------------------------------------------------------------------
    # Construction from config.yaml
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, path: str | Path = "config.yaml") -> "Indexer":
        """Build an :class:`Indexer` from *path* (a YAML config file).

        Reads ``indexing.db_path`` and optionally ``indexing.batch_size``.
        The SigLIP model is constructed via :meth:`SigLIPModel.from_config`
        using the same file.

        Example::

            idx = Indexer.from_config("config.yaml")
        """
        cfg = yaml.safe_load(Path(path).read_text())
        model = SigLIPModel.from_config(path)
        idx_cfg = cfg["indexing"]
        return cls(
            model=model,
            db_path=idx_cfg["db_path"],
            batch_size=idx_cfg.get("batch_size"),
        )

    # ------------------------------------------------------------------
    # LanceDB schema & table helpers
    # ------------------------------------------------------------------

    #: Name of the side table holding one calibration row per collection.
    META_TABLE = "_collection_meta"

    def _schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("id",            pa.string()),
            pa.field("collection_id", pa.string()),
            pa.field("vector",        pa.list_(pa.float32(), self.model.embedding_dim)),
            pa.field("path",          pa.string()),
            pa.field("depth_path",    pa.string()),
            # Variable-length (not fixed-size) on purpose: a fixed-size float
            # list is treated by LanceDB as a *vector* column, which would make
            # vector search ambiguous against the real ``vector`` column. The
            # 16-element invariant is enforced in ``_flatten_pose`` instead.
            pa.field("cam2world",     pa.list_(pa.float32())),
        ])

    def _meta_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("collection_id", pa.string()),
            pa.field("fx",            pa.float64()),
            pa.field("fy",            pa.float64()),
            pa.field("cx",            pa.float64()),
            pa.field("cy",            pa.float64()),
            pa.field("depth_scale",   pa.float64()),
        ])

    def _open_or_create_table(self, collection_id: str):
        """Return the LanceDB table for *collection_id*, creating it if absent."""
        if collection_id in self.db.list_tables().tables:
            return self.db.open_table(collection_id)
        return self.db.create_table(collection_id, schema=self._schema())

    def _write_meta(
        self,
        collection_id: str,
        intrinsics: dict,
        depth_scale: float,
    ) -> None:
        """Upsert the per-collection calibration row into :attr:`META_TABLE`.

        Args:
            collection_id: The collection whose calibration this describes.
            intrinsics:    Mapping with float keys ``fx``, ``fy``, ``cx``, ``cy``.
            depth_scale:   Divisor that converts raw depth units to metres.
        """
        try:
            missing = [k for k in ("fx", "fy", "cx", "cy") if k not in intrinsics]
        except TypeError as exc:
            raise ValueError(
                f"intrinsics must be a mapping with fx/fy/cx/cy; got "
                f"{type(intrinsics).__name__!r}."
            ) from exc
        if missing:
            raise ValueError(f"intrinsics is missing keys: {missing!r}")

        if self.META_TABLE in self.db.list_tables().tables:
            meta = self.db.open_table(self.META_TABLE)
        else:
            meta = self.db.create_table(self.META_TABLE, schema=self._meta_schema())

        row = {
            "collection_id": collection_id,
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "depth_scale": float(depth_scale),
        }
        (
            meta.merge_insert("collection_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([row])
        )

    # ------------------------------------------------------------------
    # Input normalisation
    # ------------------------------------------------------------------

    def _normalize(
        self,
        images: list,
        collection_id: str,
        ids: list[str] | None,
        depth_paths: list[str],
        poses: list,
    ) -> list[dict]:
        """Validate inputs and return a uniform list of record dicts.

        Each record::

            {
                "load":       Callable[[], PIL.Image.Image],  # lazy image loader
                "id":         str,                  # unique row key
                "path":       str,                  # source RGB path (or "")
                "depth_path": str,                  # paired depth-map path
                "cam2world":  list[float],          # row-major flattened 4x4 pose
            }

        Accepted *images* element types:

        - ``str`` / :class:`pathlib.Path` — image file path; loaded lazily.
        - :class:`PIL.Image.Image` — pre-loaded image.

        ``depth_paths`` and ``poses`` are **required** and must each have the
        same length as ``images`` (one depth file and one camera pose per
        frame); they are what makes a collection projectable to 3D.

        Id derivation when *ids* is ``None``:

        - Path → ``sha1("{collection_id}:{path}")`` — stable across re-runs.
        - PIL → ``sha1(png_bytes)`` — deterministic per pixel content, but
          slower; callers indexing in-memory images are encouraged to supply
          ids.
        """
        n = len(images)
        if ids is not None and len(ids) != n:
            raise ValueError(
                f"len(ids)={len(ids)} does not match len(images)={n}"
            )
        if depth_paths is None or len(depth_paths) != n:
            raise ValueError(
                f"depth_paths is required and must match len(images)={n}; "
                f"got {None if depth_paths is None else len(depth_paths)}."
            )
        if poses is None or len(poses) != n:
            raise ValueError(
                f"poses is required and must match len(images)={n}; "
                f"got {None if poses is None else len(poses)}."
            )

        records: list[dict] = []
        for i, item in enumerate(images):
            if isinstance(item, (str, Path)):
                path = str(item)
                rid: str = (
                    ids[i]
                    if ids is not None
                    else hashlib.sha1(
                        f"{collection_id}:{path}".encode()
                    ).hexdigest()
                )
                # Default-argument capture avoids late-binding closure issue
                load: Callable[[], Image.Image] = (
                    lambda p=path: Image.open(p).convert("RGB")
                )
            elif isinstance(item, Image.Image):
                path = ""
                if ids is not None:
                    rid = ids[i]
                else:
                    buf = io.BytesIO()
                    item.save(buf, format="PNG")
                    rid = hashlib.sha1(buf.getvalue()).hexdigest()
                pil_ref = item  # capture by value
                load = (
                    lambda im=pil_ref: im.convert("RGB") if im.mode != "RGB" else im
                )
            else:
                raise TypeError(
                    f"images[{i}] must be str, Path, or PIL.Image.Image; "
                    f"got {type(item).__name__!r}"
                )

            records.append({
                "load":       load,
                "id":         rid,
                "path":       path,
                "depth_path": str(depth_paths[i]),
                "cam2world":  _flatten_pose(poses[i]),
            })

        # Detect duplicate ids within this call
        seen: set[str] = set()
        dupes: list[str] = []
        for r in records:
            if r["id"] in seen:
                dupes.append(r["id"])
            else:
                seen.add(r["id"])
        if dupes:
            raise ValueError(
                f"Duplicate ids within this call: {dupes[:3]!r}"
                + (f" … (+{len(dupes) - 3} more)" if len(dupes) > 3 else "")
            )

        return records

    # ------------------------------------------------------------------
    # Core embedding loop (shared by insert and update)
    # ------------------------------------------------------------------

    def _embed_loop(
        self,
        records: list[dict],
        table,
        write_fn: Callable,
        desc: str,
        job: JobStatus,
    ) -> None:
        """Iterate over *records* in batches, embed each batch, then call *write_fn*.

        Args:
            records:  Normalised record dicts from :meth:`_normalize`.
            table:    Open LanceDB table.
            write_fn: ``write_fn(table, rows)`` — either ``table.add`` or
                      a merge-insert closure.  Called once per batch.
            desc:     tqdm progress-bar label.
            job:      Live :class:`JobStatus` to advance after each batch.
        """
        with tqdm(total=len(records), desc=desc, unit="img") as pbar:
            for start in range(0, len(records), self.batch_size):
                batch = records[start : start + self.batch_size]

                # Load images and embed (SigLIPModel.embed_images reused verbatim)
                imgs = [r["load"]() for r in batch]
                vecs = self.model.embed_images(imgs)   # (B, embedding_dim) float32

                rows = [
                    {
                        "id":            r["id"],
                        "collection_id": job.collection_id,
                        "vector":        v.tolist(),
                        "path":          r["path"],
                        "depth_path":    r["depth_path"],
                        "cam2world":     r["cam2world"],
                    }
                    for r, v in zip(batch, vecs)
                ]
                write_fn(table, rows)

                job.advance(len(batch))
                pbar.update(len(batch))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(
        self,
        images: list,
        collection_id: str,
        *,
        depth_paths: list[str],
        poses: list,
        intrinsics: dict,
        depth_scale: float,
        ids: list[str] | None = None,
        job_id: str | None = None,
    ) -> JobStatus:
        """Embed *images* and insert them as new rows into *collection_id*.

        Raises :class:`ValueError` **before** any embedding if any derived
        id already exists in the table — use :meth:`update` to overwrite.

        Args:
            images:        List of ``str``/``Path`` (file paths) or
                           ``PIL.Image.Image`` objects.
            collection_id: Name of the LanceDB table (= the room / benchmark).
            depth_paths:   One depth-map path per image (required). Paired with
                           ``poses`` and ``intrinsics`` so the collection can be
                           back-projected to 3D at query time.
            poses:         One 4x4 cam-to-world matrix per image (required).
            intrinsics:    Per-collection camera intrinsics — a mapping with
                           keys ``fx``, ``fy``, ``cx``, ``cy``. Stored once in
                           :attr:`META_TABLE`.
            depth_scale:   Per-collection divisor converting raw depth units to
                           metres (e.g. 5000 for TUM, 1000 for ScanNet).
            ids:           Optional explicit ids, one per image.  Auto-derived
                           from the path or image content when ``None``.
            job_id:        Optional caller-supplied job id (e.g. request UUID).

        Returns:
            A :class:`~src.index.status.JobStatus` with ``state="done"``.
        """
        records = self._normalize(images, collection_id, ids, depth_paths, poses)
        table   = self._open_or_create_table(collection_id)
        self._write_meta(collection_id, intrinsics, depth_scale)

        # Pre-flight conflict check (only when the table is non-empty)
        if table.count_rows() > 0:
            new_ids     = {r["id"] for r in records}
            # Load all existing ids and intersect — safe for O(10k) rows
            all_existing = set(table.to_arrow()["id"].to_pylist())
            conflicts = new_ids & all_existing
            if conflicts:
                sample = sorted(conflicts)[:3]
                raise ValueError(
                    f"insert(): {len(conflicts)} id(s) already exist in "
                    f"'{collection_id}': {sample!r}. "
                    "Call update() to upsert existing rows."
                )

        job = JobRegistry.start(collection_id=collection_id, total=len(records), job_id=job_id)
        try:
            self._embed_loop(
                records, table,
                write_fn=lambda t, rows: t.add(rows),
                desc=f"Indexing '{collection_id}'",
                job=job,
            )
            job.finish()
        except Exception as exc:
            job.fail(str(exc))
            raise

        return job

    def update(
        self,
        images: list,
        collection_id: str,
        *,
        depth_paths: list[str],
        poses: list,
        intrinsics: dict,
        depth_scale: float,
        ids: list[str] | None = None,
        job_id: str | None = None,
    ) -> JobStatus:
        """Embed *images* and upsert them into *collection_id*.

        Existing rows whose ``id`` matches are replaced; new ids are inserted.

        Args:
            images:        List of ``str``/``Path`` or ``PIL.Image.Image``.
            collection_id: LanceDB table name.
            depth_paths:   One depth-map path per image (required).
            poses:         One 4x4 cam-to-world matrix per image (required).
            intrinsics:    Per-collection intrinsics (``fx``, ``fy``, ``cx``,
                           ``cy``); upserted into :attr:`META_TABLE`.
            depth_scale:   Per-collection depth divisor (raw units → metres).
            ids:           Optional explicit ids.
            job_id:        Optional caller-supplied job id.

        Returns:
            A :class:`~src.index.status.JobStatus` with ``state="done"``.
        """
        records = self._normalize(images, collection_id, ids, depth_paths, poses)
        table   = self._open_or_create_table(collection_id)
        self._write_meta(collection_id, intrinsics, depth_scale)

        def _upsert(t, rows: list[dict]) -> None:
            (
                t.merge_insert("id")
                 .when_matched_update_all()
                 .when_not_matched_insert_all()
                 .execute(rows)
            )

        job = JobRegistry.start(collection_id=collection_id, total=len(records), job_id=job_id)
        try:
            self._embed_loop(
                records, table,
                write_fn=_upsert,
                desc=f"Updating '{collection_id}'",
                job=job,
            )
            job.finish()
        except Exception as exc:
            job.fail(str(exc))
            raise

        return job
