"""
run_grid_qm9.py — focused QM9 sweep.

Full-width QM9 (127k rows × 8 reps × 5 props × 5 N × 7 metrics) would take
hours on CPU. Instead, we run the four (rep, metric) combinations that
matter for the revision letter:

    rep              metric              story
    ---------------  ------------------  -----------------------------------
    mol2vec          mahalanobis_theta   paper's current MSI result
    rdkit_physchem   mahalanobis_theta   MSI + descriptor alternative
    rdkit_physchem   cosine              descriptor baseline (reviewer p.5)
    ecfp4_2048       tanimoto            paper's fingerprint baseline

With N = (25, 100, 250) and bootstrap 500. Merges the resulting rows into
the existing output_grid/grid_results.csv.
"""

import os
import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from benchmark_matrix import (
    DatasetCase, evaluate_dataset, RepresentationCache,
    win_rate_table, summary_by_family,
)

DATA_DIR       = "data"
BENCHMARK_DIR  = os.path.join(DATA_DIR, "benchmark")
MOL2VEC_OUTPUT = "output_benchmark"
RESULTS_DIR    = "output_grid"


def mol2vec_getter(case):
    p = f"{MOL2VEC_OUTPUT}/{case.stem}_300d_vectors.csv"
    return pd.read_csv(p).values.astype(np.float64) if os.path.exists(p) else None


QM9_CASES = [
    DatasetCase("qm9_aniline",  os.path.join(DATA_DIR, "qm9_anilin.csv"),
                ("gap", "homo", "lumo", "mu", "alpha"),
                (25, 100, 250), "QM9", "aniline"),
    DatasetCase("qm9_pyridine", os.path.join(BENCHMARK_DIR, "qm9_pyridine.csv"),
                ("gap", "homo", "lumo", "mu"),
                (25, 100, 250), "QM9", "pyridine"),
]

REPS    = ["mol2vec", "rdkit_physchem", "ecfp4_2048"]
METRICS = ["mahalanobis_theta", "cosine", "tanimoto"]


def main():
    model = Word2Vec.load("data/model_300dim.pkl")
    cache = RepresentationCache(cache_dir="output_representations")
    rng = np.random.default_rng(42)

    frames = []
    for case in QM9_CASES:
        print(f"=== {case.dataset_label}/{case.reference_label} ===")
        t0 = time.time()
        df = evaluate_dataset(
            case,
            representations=REPS,
            metrics=METRICS,
            cache=cache,
            mol2vec_model=model,
            mol2vec_matrix_getter=mol2vec_getter,
            n_boot=500,
            rng=rng,
            verbose=True,
        )
        frames.append(df)
        print(f"  -> {len(df)} rows in {(time.time()-t0)/60:.1f} min")

    df_qm9 = pd.concat(frames, ignore_index=True)
    print(f"\nQM9 total: {len(df_qm9):,} rows")

    df1 = pd.read_csv(f"{RESULTS_DIR}/grid_results.csv")
    merged = pd.concat([df1, df_qm9], ignore_index=True)
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

    print("\n--- QM9 SUBSET (the 4 key rep/metric combos) ---")
    qm9_view = (
        df_qm9.groupby(["rep", "metric"])["deviation_norm"]
        .mean().unstack("metric").round(3)
    )
    print(qm9_view.to_string())
    print("\n--- MERGED BY FAMILY (all datasets) ---")
    print(fam.to_string(index=False))
    print("\n--- MERGED TOP 10 WIN RATES ---")
    print(wr.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
