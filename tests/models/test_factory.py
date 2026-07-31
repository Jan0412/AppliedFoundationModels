"""Tests for src.models.factory — embedder selection and its LanceDB store.

The factory is the one place that decides SigLIP-vs-CLIP, and the one place
that keeps the embedder and its store in step. Both halves are asserted here;
getting them out of step is the failure that silently corrupts an index.
"""

from __future__ import annotations

import pytest
import yaml

from src.models.clip import CLIPEmbedModel
from src.models.factory import (
    DEFAULT_EMBEDDER,
    ENV_VAR,
    db_path_for,
    embedder_name,
    load_embedder,
)
from src.models.siglib import SigLIPModel


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    """Keep a developer's exported AFM_EMBEDDER out of these assertions."""
    monkeypatch.delenv(ENV_VAR, raising=False)


def _cfg(embedder=None, db_path="data/lancedb", db_paths=None) -> dict:
    models = {"embedder": embedder} if embedder is not None else {}
    indexing = {"db_path": db_path}
    if db_paths is not None:
        indexing["db_paths"] = db_paths
    return {"models": models, "indexing": indexing}


# ---------------------------------------------------------------------------
# embedder_name
# ---------------------------------------------------------------------------


def test_reads_the_name_from_config():
    assert embedder_name(_cfg("clip")) == "clip"


def test_defaults_when_config_is_silent():
    assert embedder_name(_cfg()) == DEFAULT_EMBEDDER


def test_defaults_when_models_section_is_missing():
    assert embedder_name({}) == DEFAULT_EMBEDDER


def test_env_var_overrides_config(monkeypatch):
    """AFM_EMBEDDER=clip switches a siglip config for one run."""
    monkeypatch.setenv(ENV_VAR, "clip")
    assert embedder_name(_cfg("siglip")) == "clip"


def test_explicit_argument_beats_the_env_var(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "clip")
    assert embedder_name(_cfg("clip"), "siglip") == "siglip"


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown embedder"):
        embedder_name(_cfg("dinov2"))


def test_unknown_env_value_raises(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "nonsense")
    with pytest.raises(ValueError, match="unknown embedder"):
        embedder_name(_cfg("siglip"))


# ---------------------------------------------------------------------------
# db_path_for
# ---------------------------------------------------------------------------


def test_store_follows_the_embedder():
    """The whole point: switching embedder switches the store."""
    paths = {"siglip": "data/lancedb", "clip": "data/lancedb_clip"}
    assert db_path_for(_cfg("clip", db_paths=paths)) == "data/lancedb_clip"
    assert db_path_for(_cfg("siglip", db_paths=paths)) == "data/lancedb"


def test_store_falls_back_to_db_path_when_unmapped():
    """An embedder with no db_paths entry uses the single legacy store."""
    paths = {"siglip": "data/lancedb"}
    assert db_path_for(_cfg("clip", db_path="data/fallback", db_paths=paths)) == "data/fallback"


def test_store_falls_back_when_db_paths_absent():
    assert db_path_for(_cfg("clip", db_path="data/lancedb")) == "data/lancedb"


def test_store_honours_the_env_override(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "clip")
    paths = {"siglip": "data/lancedb", "clip": "data/lancedb_clip"}
    assert db_path_for(_cfg("siglip", db_paths=paths)) == "data/lancedb_clip"


def test_store_missing_entirely_raises():
    with pytest.raises(KeyError):
        db_path_for({"models": {"embedder": "clip"}, "indexing": {}})


# ---------------------------------------------------------------------------
# load_embedder
# ---------------------------------------------------------------------------


def _write_config(tmp_path, embedder: str):
    cfg = {
        "models": {
            "embedder": embedder,
            "siglip": {"model_id": "google/siglip2-base-patch16-224",
                       "device": "cpu", "batch_size": 8},
            "clip": {"model_id": "openai/clip-vit-base-patch16",
                     "device": "cpu", "batch_size": 8},
        },
        "indexing": {"db_path": "data/lancedb"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg))
    return path


def test_load_embedder_builds_siglip(tmp_path, mock_siglip_patches):
    model = load_embedder(_write_config(tmp_path, "siglip"))
    assert isinstance(model, SigLIPModel)


def test_load_embedder_builds_clip(tmp_path, mock_clip_patches):
    model = load_embedder(_write_config(tmp_path, "clip"))
    assert isinstance(model, CLIPEmbedModel)


def test_load_embedder_passes_the_models_own_section(tmp_path, mock_clip_patches):
    """Each embedder reads its own model_id/device/batch_size, not the other's."""
    model = load_embedder(_write_config(tmp_path, "clip"))
    mock_clip_patches["AutoModel"].from_pretrained.assert_called_once_with(
        "openai/clip-vit-base-patch16", device_map="cpu"
    )
    assert model.batch_size == 8


def test_load_embedder_honours_the_env_override(tmp_path, mock_clip_patches, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "clip")
    assert isinstance(load_embedder(_write_config(tmp_path, "siglip")), CLIPEmbedModel)


def test_load_embedder_explicit_name_beats_config(tmp_path, mock_clip_patches):
    assert isinstance(load_embedder(_write_config(tmp_path, "siglip"), "clip"), CLIPEmbedModel)
