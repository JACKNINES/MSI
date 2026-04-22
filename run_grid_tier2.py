"""
run_grid_tier2.py — add Uni-Mol + hybrid results on cached cases and merge
into the grid produced by run_grid.py.
"""

import os
import time

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

from benchmark_matrix import DatasetCase, run_grid, win_rate_table, summary_by_family

BENCHMARK_DIR  = "data/benchmark"
MOL2VEC_OUTPUT = "output_benchmark"
UNIMOL_OUTPUT  = "output_unimol"
RESULTS_DIR    = "output_grid"

# Only cases that already have cached Uni-Mol vectors.
CACHED_CASES = [
    DatasetCase("qm8_aniline",       f"{BENCHMARK_DIR}/qm8_aniline.csv",
                ("E1-CC2","E2-CC2","E1-CAM"), (10,25,50,100,250),
                "QM8","aniline"),
    DatasetCase("qm8_pyridine",      f"{BENCHMARK_DIR}/qm8_pyridine.csv",
                ("E1-CC2","E2-CC2","E1-CAM"), (10,25,50,100,250),
                "QM8","pyridine"),
    DatasetCase("qm8_phenol",        f"{BENCHMARK_DIR}/qm8_phenol.csv",
                ("E1-CC2","E2-CC2","E1-CAM"), (10,25,50,100,250),
                "QM8","phenol"),
    DatasetCase("esol_caffeine",     f"{BENCHMARK_DIR}/esol_caffeine.csv",
                ("logS",), (10,25,50,100), "ESOL","caffeine"),
    DatasetCase("esol_naphthalene",  f"{BENCHMARK_DIR}/esol_naphthalene.csv",
                ("logS",), (10,25,50,100), "ESOL","naphthalene"),
    DatasetCase("freesolv_toluene",  f"{BENCHMARK_DIR}/freesolv_toluene.csv",
                ("hydration_free_energy",), (10,25,50), "FreeSolv","toluene"),
    DatasetCase("freesolv_ethanol",  f"{BENCHMARK_DIR}/freesolv_ethanol.csv",
                ("hydration_free_energy",), (10,25,50), "FreeSolv","ethanol"),
    DatasetCase("lipo_diclofenac",   f"{BENCHMARK_DIR}/lipo_diclofenac.csv",
                ("logD",), (10,25,50,100), "Lipophilicity","diclofenac"),
    DatasetCase("lipo_naproxen",     f"{BENCHMARK_DIR}/lipo_naproxen.csv",
                ("logD",), (10,25,50,100), "Lipophilicity","naproxen"),
]


def mol2vec_getter(case):
    p = f"{MOL2VEC_OUTPUT}/{case.stem}_300d_vectors.csv"
    return pd.read_csv(p).values.astype(np.float64) if os.path.exists(p) else None


def unimol_getter(case):
    p = f"{UNIMOL_OUTPUT}/{case.stem}_unimol_512d_vectors.csv"
    return pd.read_csv(p).values.astype(np.float64) if os.path.exists(p) else None


def main():
    model = Word2Vec.load("data/model_300dim.pkl")
    reps = ["unimol", "hybrid"]

    t0 = time.time()
    df2 = run_grid(
        cases=CACHED_CASES,
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

    # Merge with Tier-1 results
    df1 = pd.read_csv(f"{RESULTS_DIR}/grid_results.csv")
    merged = pd.concat([df1, df2], ignore_index=True)
    merged.to_csv(f"{RESULTS_DIR}/grid_results.csv", index=False)
    print(f"Merged grid_results.csv now has {len(merged):,} rows")

    # Rebuild summaries
    wr = win_rate_table(merged)
    wr.to_csv(f"{RESULTS_DIR}/win_rate_vs_ecfp4_tanimoto.csv", index=False)

    fam = summary_by_family(merged)
    fam.to_csv(f"{RESULTS_DIR}/summary_by_family.csv", index=False)

    final = (
        merged.groupby(["rep","metric"])["deviation_norm"]
        .mean().unstack("metric").round(3)
    )
    final.to_csv(f"{RESULTS_DIR}/final_representation_x_metric.csv")

    print("\n--- TOP 20 WIN RATES (with Uni-Mol / hybrid) ---")
    print(wr.head(20).to_string(index=False))
    print("\n--- BY FAMILY ---")
    print(fam.to_string(index=False))
    print("\n--- FINAL TABLE ---")
    print(final.to_string())


if __name__ == "__main__":
    main()
