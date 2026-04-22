"""
run_grid.py — headless runner for benchmark_representations.ipynb.

Writes the same artefacts (grid_results.csv, win_rate, summaries, heatmaps)
directly to ``output_grid/`` so we can execute the sweep from a terminal /
background task without Jupyter.
"""

from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec

from benchmark_matrix import (
    DatasetCase, run_grid, win_rate_table, summary_by_family,
)
from representations import ALL_REPRESENTATIONS
from similarity_measures import COMPATIBLE_METRICS


DATA_DIR       = "data"
BENCHMARK_DIR  = os.path.join(DATA_DIR, "benchmark")
MOL2VEC_OUTPUT = "output_benchmark"
UNIMOL_OUTPUT  = "output_unimol"
REP_CACHE      = "output_representations"
RESULTS_DIR    = "output_grid"
MODEL_PATH     = os.path.join(DATA_DIR, "model_300dim.pkl")


def build_cases(include_qm9: bool = False):
    cases = [
        DatasetCase("qm9_aniline", os.path.join(DATA_DIR, "qm9_anilin.csv"),
                    ("gap", "homo", "lumo", "mu", "alpha"),
                    (10, 25, 50, 100, 250), "QM9", "aniline"),
        DatasetCase("qm9_pyridine", os.path.join(BENCHMARK_DIR, "qm9_pyridine.csv"),
                    ("gap", "homo", "lumo", "mu"),
                    (10, 25, 50, 100, 250), "QM9", "pyridine"),
        DatasetCase("qm8_aniline", os.path.join(BENCHMARK_DIR, "qm8_aniline.csv"),
                    ("E1-CC2", "E2-CC2", "E1-CAM"),
                    (10, 25, 50, 100, 250), "QM8", "aniline"),
        DatasetCase("qm8_pyridine", os.path.join(BENCHMARK_DIR, "qm8_pyridine.csv"),
                    ("E1-CC2", "E2-CC2", "E1-CAM"),
                    (10, 25, 50, 100, 250), "QM8", "pyridine"),
        DatasetCase("qm8_phenol", os.path.join(BENCHMARK_DIR, "qm8_phenol.csv"),
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
    if not include_qm9:
        cases = [c for c in cases if c.dataset_label != "QM9"]
    return cases


def mol2vec_getter(case):
    p = os.path.join(MOL2VEC_OUTPUT, f"{case.stem}_300d_vectors.csv")
    if os.path.exists(p):
        return pd.read_csv(p).values.astype(np.float64)
    return None


def unimol_getter(case):
    p = os.path.join(UNIMOL_OUTPUT, f"{case.stem}_unimol_512d_vectors.csv")
    if os.path.exists(p):
        return pd.read_csv(p).values.astype(np.float64)
    return None


def family_of(rep):
    if rep in ("rdkit_physchem",):
        return "descriptor"
    if rep in ("mol2vec", "unimol", "hybrid"):
        return "embedding"
    return "fingerprint"


def render_heatmaps(results: pd.DataFrame):
    pivot = (
        results
        .groupby(["rep", "metric"])["deviation_norm"]
        .mean()
        .unstack("metric")
    )
    mask = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    for rep in pivot.index:
        allowed = set(COMPATIBLE_METRICS[family_of(rep)])
        for m in pivot.columns:
            if m not in allowed:
                mask.loc[rep, m] = True

    sns.set_theme(style="white", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", mask=mask,
                cmap="YlGn_r", cbar_kws={"label": "mean norm deviation"},
                vmin=0.3, vmax=1.1, ax=ax)
    ax.set_title("Mean normalised property deviation — lower is better")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "heatmap_norm_deviation.png"), dpi=200)
    plt.close(fig)

    # By domain
    groups = {
        "electronic (QM8, QM9)": ["QM8", "QM9"],
        "experimental (ESOL, FreeSolv, Lipo)":
            ["ESOL", "FreeSolv", "Lipophilicity"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, (title, labels) in zip(axes, groups.items()):
        subset = results[results["dataset"].isin(labels)]
        if subset.empty:
            ax.set_title(f"{title} — no data")
            ax.axis("off")
            continue
        piv = (subset.groupby(["rep", "metric"])["deviation_norm"]
               .mean().unstack("metric"))
        sub_mask = pd.DataFrame(False, index=piv.index, columns=piv.columns)
        for rep in piv.index:
            allowed = set(COMPATIBLE_METRICS[family_of(rep)])
            for m in piv.columns:
                if m not in allowed:
                    sub_mask.loc[rep, m] = True
        sns.heatmap(piv, annot=True, fmt=".3f", mask=sub_mask,
                    cmap="YlGn_r", cbar_kws={"label": "mean norm"},
                    vmin=0.3, vmax=1.1, ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "heatmap_by_domain.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier2", action="store_true",
                        help="include unimol + hybrid (requires cached vectors)")
    parser.add_argument("--qm9", action="store_true",
                        help="include QM9 cases (~127k molecules each)")
    parser.add_argument("--boot", type=int, default=1000,
                        help="bootstrap iterations (default 1000)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    reps = [
        "ecfp4_1024", "ecfp4_2048", "ecfp4_4096", "ecfp6_2048",
        "maccs", "rdkit_fp", "rdkit_physchem", "mol2vec",
    ]
    if args.tier2:
        reps += ["unimol", "hybrid"]

    cases = build_cases(include_qm9=args.qm9)
    print(f"[run_grid] {len(cases)} cases, {len(reps)} representations, "
          f"n_boot={args.boot}")
    for c in cases:
        n = sum(1 for _ in open(c.filename)) - 1
        print(f"  {c.dataset_label:>14s}/{c.reference_label:<14s} "
              f"({n:>7,} mol)")
    print()

    print(f"[run_grid] loading mol2vec model…")
    t0 = time.time()
    model = Word2Vec.load(MODEL_PATH)
    print(f"  loaded in {time.time() - t0:.1f}s "
          f"(vocab {len(model.wv):,}, dim {model.wv.vector_size})")

    t0 = time.time()
    results = run_grid(
        cases=cases,
        representations=reps,
        cache_dir=REP_CACHE,
        mol2vec_model=model,
        mol2vec_matrix_getter=mol2vec_getter,
        unimol_matrix_getter=unimol_getter,
        n_boot=args.boot,
        seed=42,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n[run_grid] TOTAL: {len(results):,} rows in {elapsed/60:.1f} min")

    results.to_csv(os.path.join(RESULTS_DIR, "grid_results.csv"), index=False)
    print(f"  saved grid_results.csv")

    wr = win_rate_table(results)
    wr.to_csv(os.path.join(RESULTS_DIR, "win_rate_vs_ecfp4_tanimoto.csv"), index=False)
    print(f"  saved win_rate_vs_ecfp4_tanimoto.csv")

    fam = summary_by_family(results)
    fam.to_csv(os.path.join(RESULTS_DIR, "summary_by_family.csv"), index=False)
    print(f"  saved summary_by_family.csv")

    pivot = (
        results.groupby(["rep", "metric"])["deviation_norm"]
        .mean().unstack("metric").round(3)
    )
    pivot.to_csv(os.path.join(RESULTS_DIR, "final_representation_x_metric.csv"))
    print(f"  saved final_representation_x_metric.csv")

    render_heatmaps(results)
    print(f"  saved heatmaps to {RESULTS_DIR}/")

    print("\n--- TOP 15 WIN RATES vs ECFP4/Tanimoto ---")
    print(wr.head(15).to_string(index=False))
    print("\n--- BY FAMILY ---")
    print(fam.to_string(index=False))
    print("\n--- FINAL TABLE (mean norm deviation) ---")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
