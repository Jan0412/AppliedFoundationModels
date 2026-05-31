"""Tests for src.models.base.BaseModel."""
from __future__ import annotations

import pytest
import yaml

from src.models.base import BaseModel


# ---------------------------------------------------------------------------
# Concrete subclasses used only in this test module
# ---------------------------------------------------------------------------


class _SigLIPConcreteModel(BaseModel):
    """Minimal concrete subclass that records constructor kwargs."""

    _config_key = "siglip"

    def __init__(self, **kwargs):
        self.received_kwargs = kwargs

    def invoke(self, input, config=None, **kwargs):
        return input


class _SAMConcreteModel(BaseModel):
    """Minimal concrete subclass with the SAM config key."""

    _config_key = "sam"

    def __init__(self, **kwargs):
        self.received_kwargs = kwargs

    def invoke(self, input, config=None, **kwargs):
        return input


class _NoKeyModel(BaseModel):
    """Subclass that deliberately omits ``_config_key``."""

    def invoke(self, input, config=None, **kwargs):
        return input


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_base_model_is_abstract():
    """BaseModel cannot be instantiated directly (abstract invoke)."""
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


def test_from_config_reads_yaml_and_passes_kwargs(tmp_config):
    """from_config loads the YAML and forwards the matching section to __init__."""
    model = _SigLIPConcreteModel.from_config(tmp_config)
    assert model.received_kwargs == {
        "model_id": "google/siglip2-base-patch16-224",
        "device": "auto",
        "batch_size": 64,
    }


def test_from_config_uses_subclass_config_key(tmp_config):
    """from_config uses _config_key of the calling subclass."""
    model = _SAMConcreteModel.from_config(tmp_config)
    assert model.received_kwargs == {
        "model_id": "facebook/sam3",
        "device": "auto",
        "threshold": 0.5,
        "mask_threshold": 0.5,
    }


def test_from_config_missing_section_raises_key_error(tmp_path):
    """from_config raises KeyError when the expected section is missing."""
    minimal_config = {"models": {"other_model": {}}}
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(minimal_config))

    with pytest.raises(KeyError):
        _SigLIPConcreteModel.from_config(config_file)  # expects "siglip" section


def test_from_config_no_config_key_raises_not_implemented():
    """from_config raises NotImplementedError when _config_key is None."""
    with pytest.raises(NotImplementedError, match="_config_key"):
        _NoKeyModel.from_config("config.yaml")


def test_concrete_subclass_is_instantiable_and_invokable():
    """A properly defined subclass can be constructed and invoked."""
    model = _SigLIPConcreteModel(model_id="x", device="cpu")
    assert model.invoke("hello") == "hello"


def test_concrete_subclass_is_runnable_instance():
    """Subclasses of BaseModel are instances of langchain Runnable."""
    from langchain_core.runnables import Runnable

    model = _SigLIPConcreteModel(model_id="x")
    assert isinstance(model, Runnable)
