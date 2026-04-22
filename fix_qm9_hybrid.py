"""
fix_qm9_hybrid.py — repair mol2vec + hybrid results for the QM9 cases.

Why: the cached ``output_benchmark/qm9_*_300d_vectors.csv`` matrices were
generated with vocabulary filtering (drops ~5% of rows whose mol2vec
identifiers aren't in the Word2Vec vocabulary). Row N of the cached
matrix is therefore the N-th *valid* molecule, not the N-th original
molecule — so when ``evaluate_dataset`` does ``prop_values[top_idx]`` it
looks up the wrong molecules. On top of that, the hybrid concat refused
to run because mol2vec (127 537 rows) and Uni-Mol (133 885 rows) don't
line up.

Fix:

    1. Regenerate mol2vec at full dataset length using
       ``representations.generate_representation("mol2vec", ...)`` — that
       function keeps a row-per-molecule matrix with NaN for vocabulary
       misses.
    2. Impute NaN with 0 (post-standardisation, column mean ≈ 0, so this
       is statistically neutral) so the downstream covariance stays valid.
    3. Build hybrid = concat(normalise(mol2vec), normalise(Uni-Mol)).
    4. Drop the old QM9 rows (rep ∈ {mol2vec, hybrid}) from
       ``output_grid/grid_results.csv`` and replace them with freshly
       evaluated ones.

Run:
    python3 fix_qm9_hybrid.py
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from benchmark_matrix import (
    DatasetCase, evaluate_dataset, RepresentationCache,
    win_rate_table, summary_by_family,
)
from representations import generate_representation
from unimol_msi import build_hybrid_vectors


DATA_DIR      = "data"
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmark")
UNIMOL_OUTPUT = "output_unimol"
REP_CACHE     = "output_representations"
RESULTS_DIR   = "output_grid"


QM9_CASES = [
    DatasetCase("qm9_aniline",  os.path.join(DATA_DIR, "qm9_anilin.csv"),
                ("gap", "homo", "lumo", "mu", "alpha"),
                (10, 25, 50, 100, 250), "QM9", "aniline"),
    DatasetCase("qm9_pyridine", os.path.join(BENCHMARK_DIR, "qm9_pyridine.csv"),
                ("gap", "homo", "lumo", "mu"),
                (10, 25, 50, 100, 250), "QM9", "pyridine"),
]


def load_unimol(case: DatasetCase) -> np.ndarray:
    p = os.path.join(UNIMOL_OUTPUT, f"{case.stem}_unimol_512d_vectors.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Uni-Mol vectors missing: {p}")
    return pd.read_csv(p).values.astype(np.float64)


def rebuild_mol2vec_and_hybrid(case: DatasetCase, model: Word2Vec):
    df = pd.read_csv(case.filename)
    smiles = df["smiles"].astype(str).tolist()
    N = len(smiles)
    print(f"\n=== {case.stem}: {N:,} molecules ===")

    # --- 1. mol2vec full-length (NaN for vocab misses) --------------------
    t0 = time.time()
    m2v = generate_representation(smiles, "mol2vec", mol2vec_model=model)
    nan_rows = np.isnan(m2v).any(axis=1)
    print(f"  mol2vec {m2v.shape} with {int(nan_rows.sum()):,} NaN rows "
          f"({nan_rows.mean():.1%}) in {time.time()-t0:.1f}s")

    # --- 2. impute NaN → 0 ------------------------------------------------
    m2v_clean = np.where(np.isnan(m2v), 0.0, m2v)

    # --- 3. Uni-Mol -------------------------------------------------------
    um = load_unimol(case)
    print(f"  unimol  {um.shape}")
    if um.shape[0] != m2v_clean.shape[0]:
        raise RuntimeError(
            f"Still mismatched after rebuild: m2v={m2v_clean.shape}, "
            f"unimol={um.shape}"
        )

    # --- 4. hybrid = L2-normalised concat --------------------------------
    hybrid = build_hybrid_vectors(m2v_clean, um, normalize=True)
    print(f"  hybrid  {hybrid.shape}")

    # --- 5. overwrite the cached .npy matrices ---------------------------
    os.makedirs(REP_CACHE, exist_ok=True)
    np.save(os.path.join(REP_CACHE, f"{case.stem}__mol2vec.npy"), m2v_clean)
    np.save(os.path.join(REP_CACHE, f"{case.stem}__hybrid.npy"),  hybrid)
    print(f"  cached {case.stem}__mol2vec.npy and {case.stem}__hybrid.npy")


def prebuilt_matrix_getter(rep_type: str):
    """Return a matrix getter that hits the freshly-rebuilt npy cache."""
    def _get(case):
        p = os.path.join(REP_CACHE, f"{case.stem}__{rep_type}.npy")
        if os.path.exists(p):
            return np.load(p)
        return None
    return _get


def main():
    # ---- 1. Rebuild matrices --------------------------------------------
    print("=" * 70)
    print("Step 1/3: rebuild mol2vec + hybrid matrices for QM9")
    print("=" * 70)
    model = Word2Vec.load(os.path.join(DATA_DIR, "model_300dim.pkl"))
    for case in QM9_CASES:
        rebuild_mol2vec_and_hybrid(case, model)

    # ---- 2. Re-evaluate mol2vec + hybrid on QM9 -------------------------
    print("\n" + "=" * 70)
    print("Step 2/3: re-evaluate mol2vec + hybrid on QM9")
    print("=" * 70)

    cache = RepresentationCache(cache_dir=REP_CACHE)
    rng = np.random.default_rng(42)

    m2v_getter = prebuilt_matrix_getter("mol2vec")
    um_getter  = prebuilt_matrix_getter("unimol")   # falls back to output_unimol via Tier-2 ran; we prefer .npy cache

    # Ensure unimol .npy is available too (the Tier-2 run already cached it)
    for case in QM9_CASES:
        um_npy = os.path.join(REP_CACHE, f"{case.stem}__unimol.npy")
        if not os.path.exists(um_npy):
            um = load_unimol(case)
            np.save(um_npy, um)
            print(f"  cached {case.stem}__unimol.npy ({um.shape})")

    frames = []
    for case in QM9_CASES:
        print(f"\n--- {case.stem} ---")
        t0 = time.time()
        df = evaluate_dataset(
            case,
            representations=["mol2vec", "hybrid"],
            cache=cache,
            mol2vec_model=model,
            mol2vec_matrix_getter=m2v_getter,
            unimol_matrix_getter=um_getter,
            n_boot=1000,
            rng=rng,
            verbose=True,
        )
        frames.append(df)
        print(f"  -> {len(df)} rows in {(time.time()-t0)/60:.1f} min")

    new_rows = pd.concat(frames, ignore_index=True)
    print(f"\nNew QM9 rows (mol2vec + hybrid): {len(new_rows):,}")

    # ---- 3. Merge into grid_results.csv, replacing stale QM9 rows --------
    print("\n" + "=" * 70)
    print("Step 3/3: merge into grid_results.csv")
    print("=" * 70)

    grid_csv = os.path.join(RESULTS_DIR, "grid_results.csv")
    old = pd.read_csv(grid_csv)
    stale = (
        old["case"].isin([c.name for c in QM9_CASES])
        & old["rep"].isin(["mol2vec", "hybrid"])
    )
    print(f"  existing grid: {len(old):,} rows")
    print(f"  dropping stale QM9 mol2vec+hybrid rows: {int(stale.sum()):,}")
    merged = pd.concat([old[~stale], new_rows], ignore_index=True)
    merged.to_csv(grid_csv, index=False)
    print(f"  merged grid:  {len(merged):,} rows")

    # Rebuild summaries
    wr = win_rate_table(merged)
    wr.to_csv(os.path.join(RESULTS_DIR, "win_rate_vs_ecfp4_tanimoto.csv"), index=False)
    fam = summary_by_family(merged)
    fam.to_csv(os.path.join(RESULTS_DIR, "summary_by_family.csv"), index=False)
    final = (
        merged.groupby(["rep","metric"])["deviation_norm"]
        .mean().unstack("metric").round(3)
    )
    final.to_csv(os.path.join(RESULTS_DIR, "final_representation_x_metric.csv"))

    print("\n--- TOP 15 WIN RATES (repaired grid) ---")
    print(wr.head(15).to_string(index=False))
    print("\n--- BY FAMILY (repaired) ---")
    print(fam.to_string(index=False))
    print("\n--- FINAL TABLE (repaired) ---")
    print(final.to_string())


if __name__ == "__main__":
    main()
