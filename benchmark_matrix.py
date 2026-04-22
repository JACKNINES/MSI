"""
benchmark_matrix.py — orchestration for the representation × metric benchmark
==============================================================================

Runs the "does MSI pick molecules that really have similar properties?" test
(same ground-truth logic as ``benchmark.ipynb``) across every combination of:

    Representation   ∈  ECFP{2,6}, ECFP4×{1k,2k,4k}, MACCS, RDKit path FP,
                        RDKit physchem descriptors,
                        Mol2Vec 300D, Uni-Mol 512D, Hybrid 812D

    Metric           ∈  Tanimoto, Tversky (α,β) asymmetric, Dice, Cosine,
                        Mahalanobis distance d_M, Mahalanobis angle θ_M

for a list of (dataset, reference-molecule) pairs. It caches each
representation matrix on disk so re-running a new metric over the same
representation is cheap.

This module, together with ``representations.py`` and ``similarity_measures.py``,
answers Reviewer 1's methodological extensions:

    Point 2 — Tversky is exposed as an explicit metric.
    Point 3 — The grid makes the "representation effect" visible as a
              dimension of the results table.
    Point 5 — Every family is benchmarked (fingerprint, descriptor, embedding).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from representations import (
    ALL_REPRESENTATIONS,
    FINGERPRINT_TYPES,
    DESCRIPTOR_TYPES,
    EMBEDDING_TYPES,
    generate_representation,
)
from similarity_measures import (
    COMPATIBLE_METRICS,
    METRIC_ORIENTATION,
    similarity_vs_reference,
    topn_indices,
)


# =============================================================================
# Dataset configuration
# =============================================================================

@dataclass
class DatasetCase:
    name: str                                   # short id, e.g. "qm8_aniline"
    filename: str                               # path to CSV with ref in row 0
    properties: Sequence[str]                   # column names for ground-truth
    n_values: Sequence[int] = (10, 25, 50, 100) # top-N cutoffs to evaluate
    dataset_label: str = ""
    reference_label: str = ""

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.filename))[0]


# =============================================================================
# Representation cache
# =============================================================================

def _family_of(rep_type: str) -> str:
    if rep_type in FINGERPRINT_TYPES:
        return "fingerprint"
    if rep_type in DESCRIPTOR_TYPES:
        return "descriptor"
    if rep_type in EMBEDDING_TYPES:
        return "embedding"
    raise KeyError(rep_type)


class RepresentationCache:
    """
    Disk-backed cache keyed by (dataset stem, rep_type).

    Each representation matrix is saved once as a ``.npy`` file so
    subsequent metric sweeps over the same representation are O(N·D)
    memory loads instead of recomputation.
    """

    def __init__(self, cache_dir: str = "output_representations"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, dataset_stem: str, rep_type: str) -> str:
        return os.path.join(self.cache_dir, f"{dataset_stem}__{rep_type}.npy")

    def get(
        self,
        dataset_stem: str,
        rep_type: str,
        smiles_list: List[str],
        **gen_kwargs,
    ) -> np.ndarray:
        path = self._path(dataset_stem, rep_type)
        if os.path.exists(path):
            mat = np.load(path, allow_pickle=False)
            if mat.shape[0] == len(smiles_list):
                return mat
            # Shape mismatch → dataset changed under us; regenerate.
        mat = generate_representation(smiles_list, rep_type, **gen_kwargs)
        np.save(path, mat)
        return mat


# =============================================================================
# Property-proximity test
# =============================================================================

def property_deviation(
    property_values: np.ndarray,
    ref_value: float,
    indices: np.ndarray,
) -> float:
    """Mean |property - ref| over the rows selected by ``indices``."""
    vals = property_values[indices]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(np.abs(vals - ref_value)))


def bootstrap_random_deviation(
    property_values: np.ndarray,
    ref_value: float,
    n: int,
    n_boot: int = 10_000,
    exclude_idx: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    ``n_boot`` random samples of size ``n`` from ``property_values``
    (excluding the reference row). Returns an array of their
    mean-absolute-deviations vs ``ref_value``.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(property_values)
    candidate_pool = np.array([i for i in range(N) if i != exclude_idx])
    devs = np.empty(n_boot, dtype=np.float64)
    finite_mask = np.isfinite(property_values)
    for i in range(n_boot):
        pick = rng.choice(candidate_pool, size=n, replace=False)
        vals = property_values[pick]
        vals = vals[finite_mask[pick]]
        if vals.size == 0:
            devs[i] = np.nan
        else:
            devs[i] = float(np.mean(np.abs(vals - ref_value)))
    return devs


# =============================================================================
# Per-dataset grid evaluation
# =============================================================================

def evaluate_dataset(
    case: DatasetCase,
    representations: Sequence[str],
    metrics: Optional[Sequence[str]] = None,
    cache: Optional[RepresentationCache] = None,
    mol2vec_model=None,
    unimol_matrix_getter=None,       # optional callable: case -> (N, 512) ndarray
    mol2vec_matrix_getter=None,      # optional callable: case -> (N, 300) ndarray
    n_boot: int = 2_000,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate every (representation × compatible metric × property × N) on one
    dataset and return a tidy DataFrame of results.

    Parameters
    ----------
    case : DatasetCase
        Dataset + reference molecule configuration.
    representations : sequence of str
        Representation keys from ``ALL_REPRESENTATIONS``.
    metrics : sequence of str, optional
        Subset of ``METRIC_ORIENTATION`` keys. If None, every metric listed
        in ``COMPATIBLE_METRICS`` for the representation family is used.
    cache : RepresentationCache, optional
        If provided, matrices are loaded/saved through it.
    mol2vec_model : gensim Word2Vec, optional
        Required to build the mol2vec / hybrid representations on-the-fly.
        Skip if ``mol2vec_matrix_getter`` returns the matrix.
    unimol_matrix_getter, mol2vec_matrix_getter : callable, optional
        ``f(case) -> ndarray`` — lets the caller reuse matrices already
        produced by ``benchmark.ipynb`` / ``benchmark_unimol.ipynb``.
    n_boot : int
        Bootstrap iterations for the random-baseline p-value.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if cache is None:
        cache = RepresentationCache()

    df = pd.read_csv(case.filename)
    smiles_list = df["smiles"].astype(str).tolist()

    # --- Alignment subset ------------------------------------------------
    # Mol2Vec-family caches (mol2vec, unimol, hybrid) were produced after
    # mol2vec's vocabulary filter, so they have FEWER rows than the raw CSV.
    # If we naively index property_values with a top-N from those reduced
    # matrices, we look up the wrong molecule. Here we compute a
    # `kept_indices` array (positions in the raw CSV that survive mol2vec's
    # filter) so every rep can be evaluated on a matched row subset.
    kept_indices = np.arange(len(smiles_list))
    aligned_path = os.path.join("output_benchmark", f"{case.stem}_2d_vectors.csv")
    if os.path.exists(aligned_path):
        try:
            aligned_smiles = pd.read_csv(aligned_path)["smiles"].astype(str).tolist()
            if 0 < len(aligned_smiles) < len(smiles_list):
                j = 0
                kept = []
                for i, s in enumerate(smiles_list):
                    if j < len(aligned_smiles) and s == aligned_smiles[j]:
                        kept.append(i)
                        j += 1
                if j == len(aligned_smiles):
                    kept_indices = np.asarray(kept, dtype=np.int64)
                    if verbose:
                        print(f"  [{case.stem}] aligned subset: "
                              f"{len(kept_indices):,}/{len(smiles_list):,} rows")
        except Exception as exc:
            if verbose:
                print(f"  [{case.stem}] alignment skipped: {exc}")

    # --- Build / load every requested representation ---------------------
    rep_mats: Dict[str, np.ndarray] = {}
    rep_times: Dict[str, float] = {}
    for rep in representations:
        t0 = time.time()

        kwargs = {}
        if rep in ("mol2vec", "hybrid"):
            m2v = mol2vec_matrix_getter(case) if mol2vec_matrix_getter else None
            if m2v is not None:
                kwargs["mol2vec_matrix"] = m2v
            elif mol2vec_model is not None:
                kwargs["mol2vec_model"] = mol2vec_model
        if rep in ("unimol", "hybrid"):
            um = unimol_matrix_getter(case) if unimol_matrix_getter else None
            if um is not None:
                kwargs["unimol_matrix"] = um
            else:
                kwargs["unimol_input_path"] = case.filename

        try:
            mat = cache.get(case.stem, rep, smiles_list, **kwargs)
        except Exception as exc:
            if verbose:
                print(f"  [{case.stem}] SKIP {rep}: {exc}")
            continue

        # Align the matrix to the common kept-indices subset. Reps that were
        # built on the full smiles list get indexed; reps already restricted
        # to the mol2vec subset are assumed to match `len(kept_indices)`.
        if mat.shape[0] == len(smiles_list) and len(kept_indices) < len(smiles_list):
            mat = mat[kept_indices]
        elif mat.shape[0] != len(kept_indices) and mat.shape[0] != len(smiles_list):
            if verbose:
                print(f"  [{case.stem}] SKIP {rep}: shape {mat.shape} "
                      f"matches neither full ({len(smiles_list)}) nor subset "
                      f"({len(kept_indices)})")
            continue

        rep_mats[rep] = mat
        rep_times[rep] = time.time() - t0
        if verbose:
            print(f"  [{case.stem}] {rep:<16s} {mat.shape} "
                  f"dtype={mat.dtype} in {rep_times[rep]:.1f}s")

    # --- Evaluate grid ----------------------------------------------------
    records: List[Dict] = []
    for prop in case.properties:
        if prop not in df.columns:
            if verbose:
                print(f"  [{case.stem}] WARN: property '{prop}' missing; skipping")
            continue
        prop_values = df[prop].to_numpy(dtype=np.float64)[kept_indices]
        ref_value = float(prop_values[0])
        if not np.isfinite(ref_value):
            if verbose:
                print(f"  [{case.stem}] WARN: ref {prop} is NaN; skipping")
            continue

        # Pre-compute bootstrap distributions (one per N)
        boot_devs: Dict[int, np.ndarray] = {}
        for n in case.n_values:
            boot_devs[n] = bootstrap_random_deviation(
                prop_values, ref_value, n=n, n_boot=n_boot, exclude_idx=0, rng=rng
            )

        # For every (rep, metric), compute top-N deviations
        for rep, mat in rep_mats.items():
            fam = _family_of(rep)
            candidate_metrics = metrics if metrics is not None else COMPATIBLE_METRICS[fam]
            for metric in candidate_metrics:
                if metric not in COMPATIBLE_METRICS[fam]:
                    continue
                try:
                    scores = similarity_vs_reference(
                        mat, ref_idx=0, metric=metric
                    )
                except Exception as exc:
                    if verbose:
                        print(f"  [{case.stem}] {rep}/{metric} ERROR: {exc}")
                    continue

                for n in case.n_values:
                    top = topn_indices(scores, metric, n=n, exclude_ref=0)
                    dev = property_deviation(prop_values, ref_value, top)
                    boot = boot_devs[n]
                    # p = Pr(random_dev ≤ observed_dev)
                    # lower-tail p-value: chance of doing this well by luck
                    finite_boot = boot[np.isfinite(boot)]
                    if finite_boot.size == 0 or not np.isfinite(dev):
                        p_val = np.nan
                        norm = np.nan
                    else:
                        p_val = float(np.mean(finite_boot <= dev))
                        mean_boot = float(finite_boot.mean())
                        norm = dev / mean_boot if mean_boot > 0 else np.nan

                    records.append({
                        "dataset":   case.dataset_label or case.name,
                        "reference": case.reference_label or case.name,
                        "case":      case.name,
                        "property":  prop,
                        "rep":       rep,
                        "rep_family": fam,
                        "metric":    metric,
                        "n":         n,
                        "deviation": dev,
                        "deviation_norm": norm,  # < 1 → better than random
                        "p_value":   p_val,
                        "n_molecules": len(prop_values),
                    })

    return pd.DataFrame.from_records(records)


# =============================================================================
# Full-grid driver
# =============================================================================

def run_grid(
    cases: Sequence[DatasetCase],
    representations: Sequence[str],
    metrics: Optional[Sequence[str]] = None,
    cache_dir: str = "output_representations",
    mol2vec_model=None,
    unimol_matrix_getter=None,
    mol2vec_matrix_getter=None,
    n_boot: int = 2_000,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run ``evaluate_dataset`` across every case and concatenate results."""
    cache = RepresentationCache(cache_dir=cache_dir)
    rng = np.random.default_rng(seed)
    frames: List[pd.DataFrame] = []

    for case in cases:
        if verbose:
            print(f"\n=== {case.dataset_label}/{case.reference_label} "
                  f"[{case.stem}] ===")
        t0 = time.time()
        df = evaluate_dataset(
            case,
            representations=representations,
            metrics=metrics,
            cache=cache,
            mol2vec_model=mol2vec_model,
            unimol_matrix_getter=unimol_matrix_getter,
            mol2vec_matrix_getter=mol2vec_matrix_getter,
            n_boot=n_boot,
            rng=rng,
            verbose=verbose,
        )
        frames.append(df)
        if verbose:
            print(f"  -> {len(df)} rows in {time.time()-t0:.1f}s")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Aggregation helpers for the response-to-reviewer tables
# =============================================================================

def win_rate_table(results: pd.DataFrame, baseline: str = "ecfp4_2048/tanimoto"
                   ) -> pd.DataFrame:
    """
    For every (rep, metric), report the fraction of (case, property, N) tuples
    in which it beats a baseline (default: ECFP4/Tanimoto, the paper's current
    primary comparison).
    """
    base_rep, base_metric = baseline.split("/")
    base = results[(results.rep == base_rep) & (results.metric == base_metric)]
    base = base.set_index(["case", "property", "n"])["deviation"]

    def _score(group: pd.DataFrame) -> pd.Series:
        joined = group.set_index(["case", "property", "n"])["deviation"]
        common = joined.index.intersection(base.index)
        if len(common) == 0:
            return pd.Series({"win_rate": np.nan, "mean_norm": np.nan, "n_rows": 0})
        wins = (joined.loc[common] < base.loc[common]).mean()
        return pd.Series({
            "win_rate": float(wins),
            "mean_norm": float(group["deviation_norm"].mean(skipna=True)),
            "n_rows": int(len(common)),
        })

    out = (
        results
        .groupby(["rep", "metric"], dropna=False)
        .apply(_score, include_groups=False)
        .reset_index()
    )
    out = out.sort_values(["win_rate", "mean_norm"], ascending=[False, True])
    return out


def summary_by_family(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate performance by representation family × metric."""
    return (
        results
        .groupby(["rep_family", "metric"], dropna=False)
        .agg(
            mean_deviation=("deviation", "mean"),
            mean_norm=("deviation_norm", "mean"),
            median_p=("p_value", "median"),
            n_rows=("deviation", "size"),
        )
        .sort_values(["mean_norm"])
        .reset_index()
    )


__all__ = [
    "DatasetCase",
    "RepresentationCache",
    "property_deviation",
    "bootstrap_random_deviation",
    "evaluate_dataset",
    "run_grid",
    "win_rate_table",
    "summary_by_family",
]
