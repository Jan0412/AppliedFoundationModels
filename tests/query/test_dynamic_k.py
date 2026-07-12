"""Tests for src/query/dynamic_k.py (pure numpy, no DB)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.query.dynamic_k import dynamic_pool_size, otsu_separability, otsu_threshold


@pytest.fixture
def bimodal():
    """Two well-separated Gaussian clusters: 500 background + 500 relevant."""
    rng = np.random.default_rng(0)
    return np.concatenate([
        rng.normal(0.1, 0.02, 500),
        rng.normal(0.9, 0.02, 500),
    ]).astype(np.float32)


@pytest.fixture
def unimodal():
    """One Gaussian blob — no relevant cluster (η ≈ 2/π ≈ 0.64)."""
    rng = np.random.default_rng(1)
    return rng.normal(0.5, 0.05, 2000).astype(np.float32)


@pytest.fixture
def skewed_pool():
    """48 background + 12 relevant scores, mimicking a real query histogram."""
    return np.concatenate([
        np.linspace(0.05, 0.15, 48),
        np.linspace(0.85, 0.95, 12),
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# otsu_threshold / otsu_separability
# ---------------------------------------------------------------------------


def test_threshold_separates_the_two_modes(bimodal):
    # σ²_B is flat across the empty gap, so the threshold lands anywhere
    # strictly between the cluster centres; what matters is the split:
    # everything kept is the relevant cluster, everything dropped background.
    thr = otsu_threshold(bimodal)
    assert 0.1 < thr < 0.9
    kept = int((bimodal >= thr).sum())
    assert 450 <= kept <= 550


def test_separability_high_for_bimodal(bimodal):
    assert otsu_separability(bimodal) > 0.85


def test_separability_below_gate_for_unimodal_gaussian(unimodal):
    # A pure Gaussian yields η ≈ 2/π ≈ 0.64 — below the 0.75 uniform baseline.
    eta = otsu_separability(unimodal)
    assert 0.5 < eta < 0.75


def test_separability_zero_for_constant_scores():
    assert otsu_separability(np.full(100, 0.3, dtype=np.float32)) == 0.0


def test_separability_zero_for_a_single_score():
    """A one-frame pool has exactly zero variance — the η ratio would be 0/0."""
    assert otsu_separability(np.array([0.3], dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# dynamic_pool_size
# ---------------------------------------------------------------------------


def test_gate_fires_on_unimodal_and_falls_back_to_min_k(unimodal):
    cut = dynamic_pool_size(unimodal, min_k=7)
    assert cut.gated is True
    assert cut.k == 7


def test_tail_strategy_is_stricter_than_plain(bimodal):
    plain = dynamic_pool_size(bimodal, strategy="plain", min_k=1, max_k=10_000)
    tail = dynamic_pool_size(bimodal, strategy="tail", min_k=1, max_k=10_000)
    assert not plain.gated and not tail.gated
    assert tail.threshold >= plain.threshold
    assert tail.k <= plain.k


def test_plain_threshold_keeps_exactly_the_relevant_cluster(skewed_pool):
    cut = dynamic_pool_size(skewed_pool, strategy="plain", min_k=1, max_k=10_000)
    assert not cut.gated
    assert cut.k == 12  # the gap threshold keeps the 12 relevant scores


def test_k_matches_threshold_count_when_unclamped(bimodal):
    cut = dynamic_pool_size(bimodal, strategy="plain", min_k=1, max_k=10_000)
    assert cut.k == int((bimodal >= cut.threshold).sum())


def test_max_k_caps_the_pool(skewed_pool):
    cut = dynamic_pool_size(skewed_pool, strategy="plain", min_k=1, max_k=5)
    assert cut.k == 5


def test_min_k_floors_the_pool(skewed_pool):
    cut = dynamic_pool_size(skewed_pool, strategy="plain", min_k=20, max_k=10_000)
    assert cut.k == 20


def test_min_k_never_exceeds_population():
    sims = np.array([0.9, 0.1, 0.1], dtype=np.float32)
    cut = dynamic_pool_size(sims, min_k=10)
    assert cut.k == 3


def test_empty_input_is_gated_with_zero_k():
    cut = dynamic_pool_size(np.array([], dtype=np.float32))
    assert cut.k == 0
    assert cut.gated is True
    assert math.isnan(cut.threshold)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        dynamic_pool_size(np.array([0.1, 0.9]), strategy="bogus")
