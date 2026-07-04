"""Otsu-based dynamic pool sizing for similarity-score retrieval.

Pure-numpy port of the analysis in ``notebooks/SimilarityHistogram.ipynb``:
instead of a fixed ``top_k``, treat the query-to-frame cosine similarities as
a 1-D histogram and split it into a relevant cluster and a background blob
with Otsu's method. The number of frames above the split becomes the
retrieval pool size.

Otsu's *effectiveness metric* η = σ²_B(t*) / σ²_total guards the split: a
pure Gaussian already yields η ≈ 2/π ≈ 0.64 and a uniform distribution 0.75,
so only values above the uniform baseline indicate a genuine relevant
cluster. Below it the scores are one background blob — the queried object is
likely not in the scene — and callers should fall back to a small fixed pool.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


def _otsu_curve(scores: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray, float]:
    """Between-class variance σ²_B(t) over *bins* candidate thresholds.

    Args:
        scores: 1-D array of similarity scores.
        bins:   Histogram resolution (like pixel intensity levels 0-255).

    Returns:
        ``(bin_centers, sigma_b_sq, total_variance)``.
    """
    sims = np.asarray(scores, dtype=np.float32)

    hist, bin_edges = np.histogram(sims, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist = hist.astype(float)
    total = hist.sum()

    weight_bg = np.cumsum(hist) / total           # ω₀(t)
    weight_fg = 1.0 - weight_bg                   # ω₁(t)

    cum_mean = np.cumsum(hist * bin_centers) / total
    global_mean = cum_mean[-1]

    # Guarded divisions: np.where still evaluates the divide everywhere,
    # so use np.divide(..., where=...) to avoid 0/0 at the histogram ends.
    mean_bg = np.divide(cum_mean, weight_bg,
                        out=np.zeros_like(weight_bg), where=weight_bg > 0)
    mean_fg = np.divide(global_mean - cum_mean, weight_fg,
                        out=np.zeros_like(weight_fg), where=weight_fg > 0)

    sigma_b_sq = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return bin_centers, sigma_b_sq, float(sims.var())


def otsu_threshold(scores: np.ndarray, bins: int = 256) -> float:
    """Optimal score threshold via Otsu's method.

    Adapted from image binarization — treats the scores as a 1-D
    'intensity histogram' and returns the split maximising the
    between-class variance.

    Args:
        scores: 1-D array of similarity scores (at least one element).
        bins:   Histogram resolution.

    Returns:
        The threshold value (a bin center).
    """
    bin_centers, sigma_b_sq, _ = _otsu_curve(scores, bins)
    return float(bin_centers[np.argmax(sigma_b_sq)])


def otsu_separability(scores: np.ndarray, bins: int = 256) -> float:
    """Otsu's effectiveness metric η = σ²_B(t*) / σ²_total ∈ [0, 1].

    Calibration — η is *not* near 0 for unimodal data:

    * pure Gaussian        η ≈ 2/π ≈ 0.64  (best split of one blob)
    * uniform              η = 0.75
    * two separated modes  η → 1

    Only values above the uniform baseline indicate a genuine relevant
    cluster; below it the scores are one background blob (object likely
    not in the scene).

    Args:
        scores: 1-D array of similarity scores (at least one element).
        bins:   Histogram resolution.

    Returns:
        η, or ``0.0`` when the scores have zero variance.
    """
    _, sigma_b_sq, var_total = _otsu_curve(scores, bins)
    if var_total == 0.0:
        return 0.0
    return float(sigma_b_sq.max() / var_total)


class PoolCut(NamedTuple):
    """Result of :func:`dynamic_pool_size`.

    Attributes:
        k:            Clamped pool size (how many top rows to keep).
        threshold:    The applied similarity threshold (``nan`` for empty
                      input). When ``gated`` the plain-Otsu value is still
                      reported for observability.
        separability: Otsu η of the full score distribution.
        gated:        ``True`` when η fell below ``min_separability`` and the
                      pool fell back to ``min_k`` (object likely absent).
    """

    k: int
    threshold: float
    separability: float
    gated: bool


def dynamic_pool_size(
    similarities: np.ndarray,
    *,
    strategy: str = "tail",
    min_k: int = 10,
    max_k: int = 400,
    min_separability: float = 0.75,
) -> PoolCut:
    """Choose how many frames to keep from a similarity distribution.

    The separability gate is a distribution-*shape* test and always runs on
    the full score array; *strategy* only moves the threshold:

    * ``"tail"``  — Otsu on scores ≥ mean (default; plain Otsu splits near
      the centre of mass and keeps far too much background).
    * ``"plain"`` — classic Otsu on the full distribution.

    Args:
        similarities:     1-D array of cosine similarities, any order.
        strategy:         ``"tail"`` or ``"plain"``.
        min_k:            Pool floor, and the fallback size when the gate
                          fires.
        max_k:            Pool cap.
        min_separability: η gate; below it the threshold sits in noise and
                          the pool falls back to ``min_k``.

    Returns:
        A :class:`PoolCut`. Since callers hold rows sorted by similarity
        descending, the pool is simply the first ``k`` rows.

    Raises:
        ValueError: On an unknown *strategy*.
    """
    if strategy not in ("tail", "plain"):
        raise ValueError(
            f"dynamic_pool_size: unknown strategy {strategy!r}. "
            "Expected 'tail' or 'plain'."
        )

    sims = np.asarray(similarities, dtype=np.float32)
    n = sims.size
    if n == 0:
        return PoolCut(k=0, threshold=float("nan"), separability=0.0, gated=True)

    eta = otsu_separability(sims)
    threshold = otsu_threshold(sims)

    if eta < min_separability:
        # No distinct relevant cluster — likely "object not in scene".
        return PoolCut(
            k=min(min_k, n), threshold=threshold, separability=eta, gated=True
        )

    if strategy == "tail":
        tail = sims[sims >= sims.mean()]
        if tail.size >= 2:
            threshold = otsu_threshold(tail)

    k = int((sims >= threshold).sum())
    k = max(k, min_k)
    k = min(k, max_k, n)
    return PoolCut(k=k, threshold=threshold, separability=eta, gated=False)
