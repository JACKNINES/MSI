"""
similarity_measures.py — Unified similarity metrics for the MSI benchmark
==========================================================================

Implements every metric used in the representation × metric sweep:

    For binary fingerprints (0/1 vectors)
        tanimoto_bits(A, B)       — |A ∩ B| / |A ∪ B|        (Jaccard)
        tversky_bits(A, B, α, β)  — asymmetric generalisation of Tanimoto
                                     (α=β=1 → Tanimoto, α=β=½ → Dice)
        dice_bits(A, B)           — 2|A∩B| / (|A|+|B|)

    For continuous vectors (floats)
        cosine_continuous(u, v)                    — standard cosine similarity
        mahalanobis_distance(u, v, inv_cov)        — d_M
        mahalanobis_angle(u, v, inv_cov)           — θ_M (degrees)

    Dataset-level vectorised helpers
        similarity_vs_reference(matrix, ref_idx, metric, **kwargs)
            → returns a 1-D array of similarity/distance values for every row,
              with the appropriate "more similar = larger/smaller" orientation
              documented in the return description.

This module covers two of Reviewer 1's points simultaneously:

    Point 2 — Tversky index is included as an explicit generalisation of
              Tanimoto, placing MSI in a broader methodological context.
    Point 5 — Provides the metric axis of the representation × metric
              benchmark matrix.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# =============================================================================
# Binary fingerprint metrics
# =============================================================================

def _as_bits(x: np.ndarray) -> np.ndarray:
    """Coerce to a uint8 bit array (fast bit-count path)."""
    if x.dtype == np.uint8:
        return x
    return x.astype(np.uint8, copy=False)


def tanimoto_bits(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto/Jaccard similarity between two bit vectors, range [0, 1]."""
    a = _as_bits(a)
    b = _as_bits(b)
    intersection = int(np.bitwise_and(a, b).sum())
    union = int(np.bitwise_or(a, b).sum())
    if union == 0:
        return 0.0
    return intersection / union


def tversky_bits(a: np.ndarray, b: np.ndarray, alpha: float, beta: float) -> float:
    """
    Tversky similarity between bit vectors. Asymmetric when ``alpha != beta``.

    .. math::

        T(A, B) = \\frac{|A \\cap B|}{|A \\cap B| + \\alpha \\cdot |A - B| + \\beta \\cdot |B - A|}

    Special cases:
      - α = β = 1          → Tanimoto/Jaccard
      - α = β = ½          → Dice
      - α = 1, β = 0       → |A ∩ B| / |A|    (prototype: B contains A?)
      - α = 0, β = 1       → |A ∩ B| / |B|    (prototype: A contains B?)

    Returns 0 on both-empty vectors.
    """
    a = _as_bits(a)
    b = _as_bits(b)
    intersection = int(np.bitwise_and(a, b).sum())
    a_not_b = int(np.bitwise_and(a, np.bitwise_not(b) & 1).sum())
    b_not_a = int(np.bitwise_and(b, np.bitwise_not(a) & 1).sum())
    denom = intersection + alpha * a_not_b + beta * b_not_a
    if denom == 0:
        return 0.0
    return intersection / denom


def dice_bits(a: np.ndarray, b: np.ndarray) -> float:
    """Sørensen–Dice coefficient on bit vectors, range [0, 1]."""
    return tversky_bits(a, b, alpha=0.5, beta=0.5)


# --- Vectorised row-vs-reference versions ------------------------------------

def tanimoto_matrix(matrix: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    """
    Tanimoto similarity of every row vs a reference row.

    Uses matrix algebra for speed:
      |A ∩ B| = A · B^T
      |A|     = row sums of A
      |A ∪ B| = |A| + |B| - |A ∩ B|
    """
    M = _as_bits(matrix)
    ref = M[ref_idx].astype(np.int32)
    rows = M.astype(np.int32)
    intersection = rows @ ref                      # (N,)
    ref_sum = int(ref.sum())
    row_sums = rows.sum(axis=1)                    # (N,)
    union = row_sums + ref_sum - intersection
    safe_union = np.where(union == 0, 1, union)
    out = intersection / safe_union
    out = np.where(union == 0, 0.0, out)
    return out.astype(np.float64)


def tversky_matrix(
    matrix: np.ndarray, ref_idx: int = 0, alpha: float = 1.0, beta: float = 1.0
) -> np.ndarray:
    """Tversky similarity of every row vs a reference row."""
    M = _as_bits(matrix)
    ref = M[ref_idx].astype(np.int32)
    rows = M.astype(np.int32)
    intersection = rows @ ref                      # (N,)
    row_sums = rows.sum(axis=1)                    # |A|
    ref_sum = int(ref.sum())                       # |B|
    a_not_b = row_sums - intersection
    b_not_a = ref_sum - intersection
    denom = intersection + alpha * a_not_b + beta * b_not_a
    safe_denom = np.where(denom == 0, 1.0, denom)
    out = intersection / safe_denom
    out = np.where(denom == 0, 0.0, out)
    return out.astype(np.float64)


def dice_matrix(matrix: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    return tversky_matrix(matrix, ref_idx=ref_idx, alpha=0.5, beta=0.5)


# =============================================================================
# Continuous-vector metrics
# =============================================================================

def cosine_continuous(u: np.ndarray, v: np.ndarray) -> float:
    """Standard cosine similarity for a pair of float vectors."""
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def cosine_matrix(matrix: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    """Cosine similarity of every row vs a reference row."""
    ref = matrix[ref_idx]
    ref_norm = float(np.linalg.norm(ref))
    row_norms = np.linalg.norm(matrix, axis=1)
    safe_rows = np.where(row_norms == 0, 1.0, row_norms)
    safe_ref = max(ref_norm, 1e-12)
    return (matrix @ ref) / (safe_rows * safe_ref)


# =============================================================================
# Mahalanobis metrics (d_M and θ_M) — the core MSI quantities
# =============================================================================

def _regularized_inv_cov(
    matrix: np.ndarray, lambda_reg: float = 1e-5, standardize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ``(matrix_used, inv_cov)`` where covariance is computed on the
    (optionally standardised) matrix with Tikhonov regularisation added.

    Standardising is strongly recommended for heterogeneous descriptor or
    embedding spaces (e.g. Uni-Mol's ~50× higher variance than mol2vec).
    """
    if standardize:
        mu = matrix.mean(axis=0)
        sigma = matrix.std(axis=0, ddof=1)
        sigma = np.where(sigma == 0, 1.0, sigma)
        matrix = (matrix - mu) / sigma

    D = matrix.shape[1]
    cov = np.cov(matrix, rowvar=False)
    cov = cov + np.eye(D) * lambda_reg
    inv_cov = np.linalg.inv(cov)
    return matrix, inv_cov


def mahalanobis_distance_matrix(
    matrix: np.ndarray,
    ref_idx: int = 0,
    lambda_reg: float = 1e-5,
    standardize: bool = True,
) -> np.ndarray:
    """
    Mahalanobis distance d_M of every row to the reference row.

    Small d_M → more similar (opposite orientation to Tanimoto/cosine).
    """
    matrix, inv_cov = _regularized_inv_cov(matrix, lambda_reg, standardize)
    ref = matrix[ref_idx]
    diff = matrix - ref
    left = diff @ inv_cov
    dsq = np.einsum("ij,ij->i", diff, left)
    return np.sqrt(np.maximum(dsq, 0.0))


def mahalanobis_angle_matrix(
    matrix: np.ndarray,
    ref_idx: int = 0,
    lambda_reg: float = 1e-5,
    standardize: bool = True,
) -> np.ndarray:
    """
    Mahalanobis angle θ_M (degrees) of every row to the reference row.

    Small θ_M → more similar. Matches ``unimol_msi.compute_theta``.
    """
    matrix, inv_cov = _regularized_inv_cov(matrix, lambda_reg, standardize)
    ref = matrix[ref_idx]

    numerator = matrix @ inv_cov @ ref
    row_norm_sq = np.einsum("ij,ij->i", matrix, matrix @ inv_cov)
    ref_norm_sq = ref @ inv_cov @ ref
    denom = np.sqrt(np.maximum(row_norm_sq, 0.0)) * np.sqrt(max(ref_norm_sq, 0.0))

    safe_denom = np.where(denom == 0, 1.0, denom)
    cos_vals = np.clip(numerator / safe_denom, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_vals))
    return np.where(denom == 0, 0.0, theta)


# =============================================================================
# Unified dispatch
# =============================================================================

# For each metric, whether "more similar" corresponds to a larger or smaller
# returned value. This drives top-N selection in the benchmark notebook.
METRIC_ORIENTATION = {
    # name          : "higher_is_more_similar" (True) / "lower_is_more_similar" (False)
    "tanimoto":       True,
    "tversky_8_2":    True,   # α=0.8, β=0.2 — asymmetric "reference contains query"
    "tversky_2_8":    True,   # α=0.2, β=0.8 — asymmetric "query contains reference"
    "dice":           True,
    "cosine":         True,
    "mahalanobis_d":  False,
    "mahalanobis_theta": False,
}


def similarity_vs_reference(
    matrix: np.ndarray,
    ref_idx: int = 0,
    metric: str = "tanimoto",
    lambda_reg: float = 1e-5,
    standardize: bool = True,
) -> np.ndarray:
    """
    Compute a similarity/distance array for every row vs ``ref_idx``.

    The caller must combine this output with ``METRIC_ORIENTATION[metric]``
    to pick the top-N most similar rows.

    Allowed ``metric`` values:

        For bit matrices (uint8 / 0-1):
            "tanimoto", "tversky_8_2", "tversky_2_8", "dice"

        For float matrices:
            "cosine", "mahalanobis_d", "mahalanobis_theta"

    Notes
    -----
    The function does not enforce the dtype match — passing a float
    fingerprint matrix to "tanimoto" still works (rows are coerced to bits
    by truncation) but is semantically wrong. The benchmark notebook is
    responsible for pairing compatible (representation, metric) pairs.
    """
    if metric == "tanimoto":
        return tanimoto_matrix(matrix, ref_idx)
    if metric == "tversky_8_2":
        return tversky_matrix(matrix, ref_idx, alpha=0.8, beta=0.2)
    if metric == "tversky_2_8":
        return tversky_matrix(matrix, ref_idx, alpha=0.2, beta=0.8)
    if metric == "dice":
        return dice_matrix(matrix, ref_idx)
    if metric == "cosine":
        return cosine_matrix(matrix, ref_idx)
    if metric == "mahalanobis_d":
        return mahalanobis_distance_matrix(matrix, ref_idx, lambda_reg, standardize)
    if metric == "mahalanobis_theta":
        return mahalanobis_angle_matrix(matrix, ref_idx, lambda_reg, standardize)

    raise ValueError(
        f"Unknown metric '{metric}'. "
        f"Valid: {sorted(METRIC_ORIENTATION)}"
    )


def topn_indices(
    similarity_or_distance: np.ndarray, metric: str, n: int, exclude_ref: int = 0
) -> np.ndarray:
    """
    Return indices of the top-N most similar rows under the given metric,
    excluding the reference row.

    Parameters
    ----------
    similarity_or_distance : np.ndarray, shape (N,)
        Output of :func:`similarity_vs_reference`.
    metric : str
        Must be a key of :data:`METRIC_ORIENTATION`.
    n : int
        Number of top hits to return.
    exclude_ref : int, optional
        Row index to force-exclude from the result (the reference itself).

    Returns
    -------
    np.ndarray of int
        Indices into the original matrix, length ``min(n, N-1)``, ordered
        from most → least similar.
    """
    higher_is_better = METRIC_ORIENTATION[metric]
    scores = similarity_or_distance.copy()
    if higher_is_better:
        scores[exclude_ref] = -np.inf
        order = np.argsort(-scores)
    else:
        scores[exclude_ref] = np.inf
        order = np.argsort(scores)
    return order[:n]


# =============================================================================
# Compatibility pairs — for the benchmark grid
# =============================================================================

# (representation family) → (metrics that are semantically appropriate)
COMPATIBLE_METRICS = {
    "fingerprint": ["tanimoto", "tversky_8_2", "tversky_2_8", "dice", "cosine",
                    "mahalanobis_d", "mahalanobis_theta"],
    "descriptor":  ["cosine", "mahalanobis_d", "mahalanobis_theta"],
    "embedding":   ["cosine", "mahalanobis_d", "mahalanobis_theta"],
}


__all__ = [
    "tanimoto_bits", "tversky_bits", "dice_bits",
    "tanimoto_matrix", "tversky_matrix", "dice_matrix",
    "cosine_continuous", "cosine_matrix",
    "mahalanobis_distance_matrix", "mahalanobis_angle_matrix",
    "similarity_vs_reference", "topn_indices",
    "METRIC_ORIENTATION", "COMPATIBLE_METRICS",
]
