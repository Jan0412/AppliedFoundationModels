"""Tests for src.ui.settings — the config ↔ pipeline knob mapping."""

from __future__ import annotations

import pytest

from src.ui import settings


class _Step:
    """Stand-in for a pipeline step: knobs are plain attributes, as in Search2D."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Pipeline:
    """Search2D's shape, minus the models."""

    def __init__(self):
        self.retrieve = _Step(
            mode="dynamic", min_k=50, max_k=200,
            min_separability=0.75, strategy="tail",
        )
        self.select = _Step(n_diverse=10, patch_frac=0.2)
        self.rerank = _Step(top_k_final=10)
        self.project = _Step(
            mode="cluster_single", max_instances=None, min_instance_size=50,
            min_views=1, voxel=0.01, cluster_eps=0.05, cluster_min_samples=10,
            bbox_percentile=(2.0, 98.0),
        )


def test_every_field_maps_onto_a_real_step_attribute():
    pipeline = _Pipeline()
    for field in settings.FIELDS:
        step = getattr(pipeline, field.step)
        assert hasattr(step, field.attr), f"{field.key} → {field.step}.{field.attr}"


def test_apply_settings_writes_to_the_owning_step():
    pipeline = _Pipeline()
    settings.apply_settings(pipeline, {
        "retrieval_mode": "topk",     # → retrieve.mode, not project.mode
        "min_k": 25,
        "n_diverse": 7,
        "top_k_final": 3,
        "mode": "cluster_instances",  # → project.mode
        "min_views": 4,
    })

    assert pipeline.retrieve.mode == "topk"
    assert pipeline.retrieve.min_k == 25
    assert pipeline.select.n_diverse == 7
    assert pipeline.rerank.top_k_final == 3
    assert pipeline.project.mode == "cluster_instances"
    assert pipeline.project.min_views == 4


def test_apply_settings_coerces_slider_floats_to_ints():
    pipeline = _Pipeline()
    settings.apply_settings(pipeline, {"min_views": 3.0, "top_k_final": 8.0})

    assert pipeline.project.min_views == 3
    assert isinstance(pipeline.project.min_views, int)
    assert isinstance(pipeline.rerank.top_k_final, int)


def test_zero_max_instances_becomes_none():
    pipeline = _Pipeline()
    settings.apply_settings(pipeline, {"max_instances": settings.NO_CAP})
    assert pipeline.project.max_instances is None

    settings.apply_settings(pipeline, {"max_instances": 3})
    assert pipeline.project.max_instances == 3


def test_bbox_percentile_pair_is_updated_element_wise():
    pipeline = _Pipeline()
    settings.apply_settings(pipeline, {"bbox_low": 5.0})
    assert pipeline.project.bbox_percentile == (5.0, 98.0)

    settings.apply_settings(pipeline, {"bbox_high": 95.0})
    assert pipeline.project.bbox_percentile == (5.0, 95.0)


def test_apply_settings_rejects_an_invalid_choice():
    # setattr bypasses ProjectTo3D's constructor check, so settings must guard.
    pipeline = _Pipeline()
    with pytest.raises(ValueError, match="bogus"):
        settings.apply_settings(pipeline, {"mode": "bogus"})


def test_apply_settings_ignores_unknown_keys():
    pipeline = _Pipeline()
    settings.apply_settings(pipeline, {"not_a_knob": 1})   # must not raise


def test_current_settings_round_trips_apply():
    pipeline = _Pipeline()
    settings.apply_settings(pipeline, {"min_views": 6, "cluster_eps": 0.03})
    values = settings.current_settings(pipeline)

    assert values["min_views"] == 6
    assert values["cluster_eps"] == 0.03
    assert values["max_instances"] == settings.NO_CAP   # None reads back as 0
    assert values["bbox_low"] == 2.0 and values["bbox_high"] == 98.0


def test_defaults_from_config_reads_both_sections():
    values = settings.defaults_from_config({
        "query": {"top_k_final": 12, "min_k": 30},
        "projection": {"min_views": 5, "bbox_percentile": [1.0, 99.0]},
    })

    assert values["top_k_final"] == 12
    assert values["min_k"] == 30
    assert values["min_views"] == 5
    assert values["bbox_low"] == 1.0 and values["bbox_high"] == 99.0


def test_defaults_from_config_falls_back_when_keys_absent():
    values = settings.defaults_from_config({})
    assert set(values) == {f.key for f in settings.FIELDS}
    assert values["mode"] == "cluster_single"
    assert values["min_views"] == 1


def test_null_max_instances_in_config_reads_back_as_no_cap():
    values = settings.defaults_from_config({"projection": {"max_instances": None}})
    assert values["max_instances"] == settings.NO_CAP
