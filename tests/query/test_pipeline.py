"""End-to-end tests for src/query/pipeline.py (Search2D)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import yaml

from src.data_model import SearchState
from src.query import (
    Detect,
    EmbedQuery,
    RerankByDetection,
    RetrieveSimilar,
    Search2D,
    SelectDiverse,
)


@pytest.fixture
def pipeline(mock_siglip_model, mock_sam_model, populated_db) -> Search2D:
    return Search2D(
        siglip=mock_siglip_model,
        detector=mock_sam_model,
        db=populated_db["db"],
    )


def test_invoke_runs_full_chain_with_kwargs(pipeline, populated_db):
    out = pipeline.invoke(
        query="anything",
        collection_id=populated_db["collection_id"],
        top_k_retrieve=3,
        top_k_final=2,
        retrieval_mode="topk",
    )

    assert out.query_embedding is not None
    assert out.retrieved is not None and len(out.retrieved) == 3
    assert out.detected is not None and len(out.detected) == 3
    assert out.results is not None and len(out.results) == 2
    # Results sorted desc by detection_score.
    scores = [r.detection_score for r in out.results]
    assert scores == sorted(scores, reverse=True)


def test_invoke_accepts_prebuilt_state(pipeline, populated_db):
    state = SearchState(
        query="anything",
        collection_id=populated_db["collection_id"],
        top_k_retrieve=3,
        top_k_final=1,
    )
    out = pipeline.invoke(state)

    assert len(out.results) == 1


def test_invoke_requires_query_or_state(pipeline):
    with pytest.raises(ValueError, match="SearchState"):
        pipeline.invoke()


def test_partial_chain_without_rerank(pipeline, mock_sam_model, populated_db):
    """The step-instances are public so callers can build a partial chain."""
    # Vary the scores so we can verify detected order is preserved (not sorted).
    score_seq = [torch.tensor([0.1]), torch.tensor([0.9]), torch.tensor([0.5])]
    mock_sam_model.invoke.side_effect = [
        {"masks": [], "boxes": torch.zeros(0, 4), "scores": s} for s in score_seq
    ]

    chain = pipeline.embed | pipeline.retrieve | pipeline.detect
    state = SearchState(
        query="anything",
        collection_id=populated_db["collection_id"],
        top_k_retrieve=3,
        retrieval_mode="topk",
    )
    out = chain.invoke(state)

    assert out.detected is not None
    assert out.results is None
    # Order matches retrieval order (which matches sim-ranking, not score-ranking).
    assert [d.detection_score for d in out.detected] == pytest.approx([0.1, 0.9, 0.5])


def test_pipeline_chain_is_runnable_sequence(pipeline):
    # The LCEL `|` composition exists as a single Runnable.
    assert hasattr(pipeline.chain, "invoke")


def test_init_wires_each_step_with_correct_type(pipeline):
    assert isinstance(pipeline.embed, EmbedQuery)
    assert isinstance(pipeline.retrieve, RetrieveSimilar)
    assert isinstance(pipeline.select, SelectDiverse)
    assert isinstance(pipeline.detect, Detect)
    assert isinstance(pipeline.rerank, RerankByDetection)


def test_init_injects_dependencies_into_steps(
    mock_siglip_model, mock_sam_model, populated_db
):
    pipeline = Search2D(
        siglip=mock_siglip_model,
        detector=mock_sam_model,
        db=populated_db["db"],
    )
    assert pipeline.embed.siglip is mock_siglip_model
    assert pipeline.detect.detector is mock_sam_model
    assert pipeline.retrieve.db is populated_db["db"]
    assert pipeline.select.db is populated_db["db"]


def test_default_retrieval_mode_is_dynamic(pipeline):
    assert pipeline.retrieve.mode == "dynamic"


def test_invoke_dynamic_mode_selects_diverse_subset(
    mock_siglip_model, mock_sam_model, db_with_sims
):
    """Dynamic pool → viewpoint selection → detector sees only n_diverse frames."""
    sims = (
        [0.85 + 0.008 * i for i in range(12)]     # relevant cluster
        + [0.05 + 0.002 * i for i in range(48)]   # background
    )
    dyn = db_with_sims(sims)
    pipeline = Search2D(
        siglip=mock_siglip_model,
        detector=mock_sam_model,
        db=dyn["db"],
        strategy="plain",
        min_k=2,
        n_diverse=4,
    )

    out = pipeline.invoke(
        query="anything", collection_id=dyn["collection_id"], top_k_final=3
    )

    assert len(out.retrieved) == 4
    assert mock_sam_model.invoke.call_count == 4
    assert len(out.results) == 3
    diag = out.retrieval_diag
    assert diag.mode == "dynamic"
    assert diag.pool_size == 12
    assert diag.n_selected == 4
    assert len(diag.selected_ids) == 4


def _write_cfg(tmp_path, db_dir, query_section=None, projection_section=None):
    cfg = {
        "models": {
            "siglip": {"model_id": "x", "device": "cpu", "batch_size": 1},
            "sam": {"model_id": "y", "device": "cpu",
                    "threshold": 0.5, "mask_threshold": 0.5},
            "grounding_dino": {"model_id": "z", "device": "cpu",
                               "box_threshold": 0.35, "text_threshold": 0.25},
        },
        "indexing": {"db_path": str(db_dir)},
    }
    if query_section is not None:
        cfg["query"] = query_section
    if projection_section is not None:
        cfg["projection"] = projection_section
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def test_from_config_default_detector_is_sam(
    tmp_path, mock_siglip_model, mock_sam_model
):
    db_dir = tmp_path / "lancedb_from_cfg"
    cfg_path = _write_cfg(tmp_path, db_dir)

    with patch(
        "src.query.pipeline.SigLIPModel.from_config",
        return_value=mock_siglip_model,
    ), patch(
        "src.query.pipeline.SAMModel.from_config",
        return_value=mock_sam_model,
    ):
        pipeline = Search2D.from_config(cfg_path)

    assert pipeline.embed.siglip is mock_siglip_model
    assert pipeline.detect.detector is mock_sam_model
    assert db_dir.exists()


def test_from_config_without_query_section_uses_defaults(
    tmp_path, mock_siglip_model, mock_sam_model
):
    cfg_path = _write_cfg(tmp_path, tmp_path / "db")

    with patch(
        "src.query.pipeline.SigLIPModel.from_config",
        return_value=mock_siglip_model,
    ), patch(
        "src.query.pipeline.SAMModel.from_config",
        return_value=mock_sam_model,
    ):
        pipeline = Search2D.from_config(cfg_path)

    assert pipeline.retrieve.mode == "dynamic"
    assert pipeline.retrieve.min_k == 10
    assert pipeline.retrieve.max_k == 400
    assert pipeline.retrieve.strategy == "tail"
    assert pipeline.select.n_diverse == 10
    # Projection defaults (no projection section present).
    assert pipeline.project.mode == "cluster_single"
    assert pipeline.project.max_instances is None
    assert pipeline.project.min_instance_size == 50


def test_from_config_reads_query_section(
    tmp_path, mock_siglip_model, mock_sam_model
):
    cfg_path = _write_cfg(
        tmp_path,
        tmp_path / "db",
        query_section={
            "retrieval_mode": "topk",
            "strategy": "plain",
            "min_k": 3,
            "max_k": 50,
            "min_separability": 0.8,
            "n_diverse": 7,
            "patch_frac": 0.3,
        },
    )

    with patch(
        "src.query.pipeline.SigLIPModel.from_config",
        return_value=mock_siglip_model,
    ), patch(
        "src.query.pipeline.SAMModel.from_config",
        return_value=mock_sam_model,
    ):
        pipeline = Search2D.from_config(cfg_path)

    assert pipeline.retrieve.mode == "topk"
    assert pipeline.retrieve.strategy == "plain"
    assert pipeline.retrieve.min_k == 3
    assert pipeline.retrieve.max_k == 50
    assert pipeline.retrieve.min_separability == 0.8
    assert pipeline.select.n_diverse == 7
    assert pipeline.select.patch_frac == 0.3


def test_from_config_reads_projection_section(
    tmp_path, mock_siglip_model, mock_sam_model
):
    cfg_path = _write_cfg(
        tmp_path,
        tmp_path / "db",
        projection_section={
            "mode": "cluster_instances",
            "max_instances": 5,
            "min_instance_size": 30,
            "cluster_eps": 0.08,
        },
    )

    with patch(
        "src.query.pipeline.SigLIPModel.from_config",
        return_value=mock_siglip_model,
    ), patch(
        "src.query.pipeline.SAMModel.from_config",
        return_value=mock_sam_model,
    ):
        pipeline = Search2D.from_config(cfg_path)

    assert pipeline.project.mode == "cluster_instances"
    assert pipeline.project.max_instances == 5
    assert pipeline.project.min_instance_size == 30
    assert pipeline.project.cluster_eps == 0.08


def test_from_config_rejects_unknown_detector(tmp_path, mock_siglip_model):
    cfg_path = _write_cfg(tmp_path, tmp_path / "db")

    with patch(
        "src.query.pipeline.SigLIPModel.from_config",
        return_value=mock_siglip_model,
    ):
        with pytest.raises(ValueError, match="unknown detector"):
            Search2D.from_config(cfg_path, detector="bogus")


def test_invoke_uses_default_top_k_kwargs(pipeline, populated_db):
    out = pipeline.invoke(
        query="anything",
        collection_id=populated_db["collection_id"],
    )
    assert len(out.retrieved) == 3
    assert len(out.results) == 3  # only 3 rows exist, so capped


def test_invoke_state_takes_precedence_over_kwargs(pipeline, populated_db):
    state = SearchState(
        query="anything",
        collection_id=populated_db["collection_id"],
        top_k_retrieve=2,
        top_k_final=1,
        retrieval_mode="topk",
    )
    out = pipeline.invoke(state, top_k_retrieve=99, top_k_final=99)
    assert len(out.retrieved) == 2
    assert len(out.results) == 1


def test_invoke_requires_collection_id_when_no_state(pipeline):
    with pytest.raises(ValueError, match="SearchState"):
        pipeline.invoke(query="anything")  # missing collection_id
