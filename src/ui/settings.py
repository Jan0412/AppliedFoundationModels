"""The tunable pipeline knobs, as data.

The ``query`` and ``projection`` sections of ``config.yaml`` are all plain
attributes on the :class:`~src.query.Search2D` step objects, and the steps
re-read them on every ``invoke``. Retuning them therefore costs an attribute
write — SigLIP and SAM stay loaded, and the chain (which holds references to
the same step objects) needs no rebuild. That is what lets the UI expose them
as live controls.

This module names each knob exactly once: which step owns it, what widget it
wants, and what range is sane. :mod:`src.ui.app` builds the Configuration panel
straight from :data:`FIELDS`, and :mod:`src.ui.service` writes the values back
onto the pipeline — neither restates the mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

#: Projection modes, mirroring :data:`src.query.project._MODES`.
PROJECTION_MODES: Tuple[str, ...] = ("cluster_single", "cluster_instances", "simple")

#: ``max_instances`` means "no cap" when unset, but viser has no nullable
#: number — the widget uses 0 and :func:`apply_settings` translates it to None.
NO_CAP = 0


@dataclass(frozen=True)
class Field:
    """One tunable knob: where it comes from, where it goes, how to show it.

    Attributes:
        key:         Stable id, used as the dict key everywhere in the UI.
        label:       Widget label.
        section:     ``config.yaml`` section the default is read from.
        step:        :class:`Search2D` attribute owning the value
                     (``"retrieve"``, ``"select"``, ``"rerank"``, ``"project"``).
        attr:        Attribute name on that step. Differs from *key* where the
                     config name is more specific than the step's own
                     (``retrieval_mode`` → ``RetrieveSimilar.mode``).
        kind:        ``"int"``, ``"float"`` or ``"choice"``.
        default:     Fallback when the config section omits the key.
        config_key:  Key inside *section*; defaults to *key*.
        pair:        Index into a 2-tuple attribute (``bbox_percentile``), or
                     ``None`` for scalars.
        options:     Allowed values, for ``"choice"``.
        min/max/inc: Slider bounds and step, for ``"int"``/``"float"``.
        zero_is_none: Widget 0 means ``None`` on the pipeline (see :data:`NO_CAP`).
        help:        One-line explanation, shown as a hint.
    """

    key: str
    label: str
    section: str
    step: str
    attr: str
    kind: str
    default: Any
    config_key: Optional[str] = None
    pair: Optional[int] = None
    options: Optional[Tuple[str, ...]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    inc: Optional[float] = None
    zero_is_none: bool = False
    help: str = ""

    @property
    def source_key(self) -> str:
        """Key to read from the config section."""
        return self.config_key or self.key


#: Retrieval / selection / rerank knobs — the ``query`` config section.
RETRIEVAL_FIELDS: Tuple[Field, ...] = (
    Field("retrieval_mode", "Retrieval mode", "query", "retrieve", "mode",
          "choice", "dynamic", options=("dynamic", "topk"),
          help="dynamic = Otsu-sized pool; topk = fixed k"),
    Field("strategy", "Otsu strategy", "query", "retrieve", "strategy",
          "choice", "tail", options=("tail", "plain"),
          help="tail is stricter; plain keeps ~60% of frames"),
    Field("top_k_final", "Results kept", "query", "rerank", "top_k_final",
          "int", 10, min=1, max=50, inc=1,
          help="frames kept after the detector rerank"),
    Field("min_k", "Pool floor", "query", "retrieve", "min_k",
          "int", 50, min=1, max=200, inc=1),
    Field("max_k", "Pool cap", "query", "retrieve", "max_k",
          "int", 200, min=10, max=1000, inc=10),
    Field("min_separability", "Min separability (η)", "query", "retrieve",
          "min_separability", "float", 0.75, min=0.0, max=1.0, inc=0.01,
          help="below this the η gate falls back to min_k"),
    Field("n_diverse", "Diverse frames", "query", "select", "n_diverse",
          "int", 10, min=1, max=50, inc=1,
          help="viewpoint-diverse frames fed to SAM"),
    Field("patch_frac", "Centre patch", "query", "select", "patch_frac",
          "float", 0.2, min=0.05, max=1.0, inc=0.05),
)

#: 3D fusion knobs — the ``projection`` config section.
PROJECTION_FIELDS: Tuple[Field, ...] = (
    Field("mode", "Mode", "projection", "project", "mode",
          "choice", "cluster_single", options=PROJECTION_MODES,
          help="cluster_instances back-projects every SAM mask per frame"),
    Field("min_instance_size", "Min instance size", "projection", "project",
          "min_instance_size", "int", 50, min=1, max=2000, inc=10,
          help="drop clusters below this many points (cluster_instances)"),
    Field("min_views", "Min views", "projection", "project", "min_views",
          "int", 1, min=1, max=20, inc=1,
          help="drop clusters backed by fewer frames; 1 = off"),
    Field("max_instances", "Max instances", "projection", "project",
          "max_instances", "int", NO_CAP, min=0, max=20, inc=1,
          zero_is_none=True, help="0 = keep every cluster"),
    Field("voxel", "Voxel (m)", "projection", "project", "voxel",
          "float", 0.01, min=0.0, max=0.1, inc=0.005,
          help="object cloud only — not the background scene cloud"),
    Field("cluster_eps", "DBSCAN eps (m)", "projection", "project",
          "cluster_eps", "float", 0.05, min=0.01, max=0.5, inc=0.01,
          help="larger merges nearby instances"),
    Field("cluster_min_samples", "DBSCAN min samples", "projection", "project",
          "cluster_min_samples", "int", 10, min=1, max=50, inc=1),
    Field("bbox_low", "BBox low %", "projection", "project", "bbox_percentile",
          "float", 2.0, config_key="bbox_percentile", pair=0,
          min=0.0, max=25.0, inc=0.5),
    Field("bbox_high", "BBox high %", "projection", "project", "bbox_percentile",
          "float", 98.0, config_key="bbox_percentile", pair=1,
          min=75.0, max=100.0, inc=0.5),
)

#: Every knob the Configuration panel exposes.
FIELDS: Tuple[Field, ...] = RETRIEVAL_FIELDS + PROJECTION_FIELDS

_BY_KEY = {f.key: f for f in FIELDS}


class _NotSet:
    """Distinguishes "absent from config" from a legitimate ``None``."""


_NOT_SET = _NotSet()


def _coerce(field: Field, value: Any) -> Any:
    """Widget value → the type the pipeline step expects."""
    if field.kind == "choice":
        if field.options and value not in field.options:
            raise ValueError(
                f"{field.key}: {value!r} is not one of {list(field.options)}."
            )
        return value
    number = int(value) if field.kind == "int" else float(value)
    if field.zero_is_none and number == NO_CAP:
        return None
    return number


def defaults_from_config(cfg: dict) -> dict:
    """Field values as configured in ``config.yaml``.

    Read from the config rather than from a live pipeline so the panel can be
    built before the models have loaded — ``SceneService`` constructs its
    pipeline lazily, and touching it here would block the UI on a model
    download.
    """
    sections = {
        "query": cfg.get("query") or {},
        "projection": cfg.get("projection") or {},
    }
    values: dict = {}
    for field in FIELDS:
        raw = sections[field.section].get(field.source_key, _NOT_SET)
        if raw is _NOT_SET:
            values[field.key] = field.default
            continue
        if field.pair is not None:
            raw = raw[field.pair]
        if field.zero_is_none and raw is None:
            raw = NO_CAP
        values[field.key] = raw
    return values


def apply_settings(pipeline, values: dict) -> None:
    """Write *values* onto the pipeline's step objects, in place.

    Unknown keys are ignored, so a stale widget can't crash a query. Choice
    fields are validated here because assigning straight to the attribute
    bypasses the step's own constructor check.
    """
    for key, value in values.items():
        field = _BY_KEY.get(key)
        if field is None:
            continue
        target = getattr(pipeline, field.step)
        coerced = _coerce(field, value)
        if field.pair is None:
            setattr(target, field.attr, coerced)
            continue
        pair = list(getattr(target, field.attr))
        pair[field.pair] = coerced
        setattr(target, field.attr, tuple(pair))


def current_settings(pipeline) -> dict:
    """Read the knobs back off a live pipeline (the inverse of :func:`apply_settings`)."""
    values: dict = {}
    for field in FIELDS:
        raw = getattr(getattr(pipeline, field.step), field.attr)
        if field.pair is not None:
            raw = raw[field.pair]
        if field.zero_is_none and raw is None:
            raw = NO_CAP
        values[field.key] = raw
    return values
