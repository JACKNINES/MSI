"""
run_grid_tier2_full.py — add Uni-Mol + hybrid over ALL cases (incl. QM9
and the new lipo/complex drugs) and merge with existing grid_results.csv.

Run this AFTER:
    1. run_on_dirac.sh has finished Step 2 (Tier-1 grid) successfully
    2. Uni-Mol vectors exist in output_unimol/ for every dataset

It only evaluates the ``unimol`` and ``hybrid`` representations so we don't
re-spend ~40 min on the Tier-1 re-evaluation.
"""

import os
import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from benchmark_matrix import (
    DatasetCase, run_grid, win_rate_table, summary_by_family,
)

DATA_DIR       = "data"
BENCHMARK_DIR  = os.path.join(DATA_DIR, "benchmark")
MOL2VEC_OUTPUT = "output_benchmark"
UNIMOL_OUTPUT  = "output_unimol"
RESULTS_DIR    = "output_grid"


def build_cases():
    cases = [
        DatasetCase("qm9_aniline",   os.path.join(DATA_DIR, "qm9_anilin.csv"),
                    ("gap", "homo", "lumo", "mu", "alpha"),
                    (10, 25, 50, 100, 250), "QM9", "aniline"),
        DatasetCase("qm9_pyridine",  os.path.join(BENCHMARK_DIR, "qm9_pyridine.csv"),
                    ("gap", "homo", "lumo", "mu"),
                    (10, 25, 50, 100, 250), "QM9", "pyridine"),
        DatasetCase("qm8_aniline",   os.path.join(BENCHMARK_DIR, "qm8_aniline.csv"),
                    ("E1-CC2", "E2-CC2", "E1-CAM"),
                    (10, 25, 50, 100, 250), "QM8", "aniline"),
        DatasetCase("qm8_pyridine",  os.path.join(BENCHMARK_DIR, "qm8_pyridine.csv"),
                    ("E1-CC2", "E2-CC2", "E1-CAM"),
                    (10, 25, 50, 100, 250), "QM8", "pyridine"),
        DatasetCase("qm8_phenol",    os.path.join(BENCHMARK_DIR, "qm8_phenol.csv"),
                    ("E1-CC2", "E2-CC2", "E1-CAM"),
                    (10, 25, 50, 100, 250), "QM8", "phenol"),
        DatasetCase("esol_caffeine", os.path.join(BENCHMARK_DIR, "esol_caffeine.csv"),
                    ("logS",), (10, 25, 50, 100), "ESOL", "caffeine"),
        DatasetCase("esol_naphthalene", os.path.join(BENCHMARK_DIR, "esol_naphthalene.csv"),
                    ("logS",), (10, 25, 50, 100), "ESOL", "naphthalene"),
        DatasetCase("freesolv_toluene", os.path.join(BENCHMARK_DIR, "freesolv_toluene.csv"),
                    ("hydration_free_energy",), (10, 25, 50), "FreeSolv", "toluene"),
        DatasetCase("freesolv_ethanol", os.path.join(BENCHMARK_DIR, "freesolv_ethanol.csv"),
                    ("hydration_free_energy",), (10, 25, 50), "FreeSolv", "ethanol"),
        DatasetCase("lipo_diclofenac", os.path.join(BENCHMARK_DIR, "lipo_diclofenac.csv"),
                    ("logD",), (10, 25, 50, 100), "Lipophilicity", "diclofenac"),
        DatasetCase("lipo_naproxen", os.path.join(BENCHMARK_DIR, "lipo_naproxen.csv"),
                    ("logD",), (10, 25, 50, 100), "Lipophilicity", "naproxen"),
    ]
    for ref in ("celecoxib", "simvastatin", "fluoxetine", "imipramine"):
        p = os.path.join(BENCHMARK_DIR, f"lipo_{ref}.csv")
        if os.path.exists(p):
            cases.append(DatasetCase(
                f"lipo_{ref}", p, ("logD",), (10, 25, 50, 100),
                "Lipophilicity", ref,
            ))
    return cases


def mol2vec_getter(case):
    p = f"{MOL2VEC_OUTPUT}/{case.stem}_300d_vectors.csv"
    return pd.read_csv(p).values.astype(np.float64) if os.path.exists(p) else None


def unimol_getter(case):
    p = f"{UNIMOL_OUTPUT}/{case.stem}_unimol_512d_vectors.csv"
    return pd.read_csv(p).values.astype(np.float64) if os.path.exists(p) else None


def main():
    model = Word2Vec.load("data/model_300dim.pkl")
    cases = build_cases()

    # Only run cases where Uni-Mol vectors are available.
    runnable = [c for c in cases if unimol_getter(c) is not None]
    skipped  = [c for c in cases if unimol_getter(c) is None]
    if skipped:
        print(f"Skipping {len(skipped)} cases without Uni-Mol vectors:")
        for c in skipped:
            print(f"  - {c.stem}")
        print()
    print(f"Running Tier-2 (unimol + hybrid) on {len(runnable)} cases")

    reps = ["unimol", "hybrid"]
    t0 = time.time()
    df2 = run_grid(
        cases=runnable,
        representations=reps,
        cache_dir="output_representations",
        mol2vec_model=model,
        mol2vec_matrix_getter=mol2vec_getter,
        unimol_matrix_getter=unimol_getter,
        n_boot=1000,
        seed=42,
        verbose=True,
    )
    print(f"\nTier-2: {len(df2):,} rows in {(time.time()-t0)/60:.1f} min")

    # Merge with existing grid
    grid_csv = os.path.join(RESULTS_DIR, "grid_results.csv")
    df1 = pd.read_csv(grid_csv)
    merged = pd.concat([df1, df2], ignore_index=True)
    merged.to_csv(grid_csv, index=False)
    print(f"Merged grid_results.csv now has {len(merged):,} rows")

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

    print("\n--- TOP 20 WIN RATES (merged grid) ---")
    print(wr.head(20).to_string(index=False))
    print("\n--- BY FAMILY (merged) ---")
    print(fam.to_string(index=False))
    print("\n--- FINAL TABLE (merged) ---")
    print(final.to_string())


if __name__ == "__main__":
    main()
