"""
run_grid_tier3.py — añade ChemBERTa + MolFormer al grid y mergea con lo existente.

Pre-requisito: corrió generate_hf_all.py y existen los CSVs 768D en output_hf/.
"""

import os
import time

import numpy as np
import pandas as pd

from benchmark_matrix import (
    DatasetCase, run_grid, win_rate_table, summary_by_family,
)

DATA_DIR       = "data"
BENCHMARK_DIR  = os.path.join(DATA_DIR, "benchmark")
HF_OUTPUT      = "output_hf"
RESULTS_DIR    = "output_grid"


def build_cases():
    cases = [
        DatasetCase("qm9_aniline",   f"{DATA_DIR}/qm9_anilin.csv",
                    ("gap","homo","lumo","mu","alpha"),
                    (10,25,50,100,250), "QM9", "aniline"),
        DatasetCase("qm9_pyridine",  f"{BENCHMARK_DIR}/qm9_pyridine.csv",
                    ("gap","homo","lumo","mu"),
                    (10,25,50,100,250), "QM9", "pyridine"),
        DatasetCase("qm8_aniline",   f"{BENCHMARK_DIR}/qm8_aniline.csv",
                    ("E1-CC2","E2-CC2","E1-CAM"),
                    (10,25,50,100,250), "QM8", "aniline"),
        DatasetCase("qm8_pyridine",  f"{BENCHMARK_DIR}/qm8_pyridine.csv",
                    ("E1-CC2","E2-CC2","E1-CAM"),
                    (10,25,50,100,250), "QM8", "pyridine"),
        DatasetCase("qm8_phenol",    f"{BENCHMARK_DIR}/qm8_phenol.csv",
                    ("E1-CC2","E2-CC2","E1-CAM"),
                    (10,25,50,100,250), "QM8", "phenol"),
        DatasetCase("esol_caffeine", f"{BENCHMARK_DIR}/esol_caffeine.csv",
                    ("logS",), (10,25,50,100), "ESOL", "caffeine"),
        DatasetCase("esol_naphthalene", f"{BENCHMARK_DIR}/esol_naphthalene.csv",
                    ("logS",), (10,25,50,100), "ESOL", "naphthalene"),
        DatasetCase("freesolv_toluene", f"{BENCHMARK_DIR}/freesolv_toluene.csv",
                    ("hydration_free_energy",), (10,25,50), "FreeSolv", "toluene"),
        DatasetCase("freesolv_ethanol", f"{BENCHMARK_DIR}/freesolv_ethanol.csv",
                    ("hydration_free_energy",), (10,25,50), "FreeSolv", "ethanol"),
        DatasetCase("lipo_diclofenac", f"{BENCHMARK_DIR}/lipo_diclofenac.csv",
                    ("logD",), (10,25,50,100), "Lipophilicity", "diclofenac"),
        DatasetCase("lipo_naproxen", f"{BENCHMARK_DIR}/lipo_naproxen.csv",
                    ("logD",), (10,25,50,100), "Lipophilicity", "naproxen"),
    ]
    for ref in ("celecoxib","simvastatin","fluoxetine","imipramine"):
        p = f"{BENCHMARK_DIR}/lipo_{ref}.csv"
        if os.path.exists(p):
            cases.append(DatasetCase(
                f"lipo_{ref}", p, ("logD",), (10,25,50,100),
                "Lipophilicity", ref,
            ))
    return cases


def hf_getter(model_key: str):
    """Devuelve una función f(case) -> matriz del CSV cacheado."""
    def _get(case):
        # El stem en el grid coincide con el filename original
        stem = case.stem
        p = f"{HF_OUTPUT}/{stem}_{model_key}_768d_vectors.csv"
        if os.path.exists(p):
            return pd.read_csv(p).values.astype(np.float64)
        return None
    return _get


# --- Patch mínimo a benchmark_matrix para reconocer los nuevos tipos ---
# El dispatch existente solo maneja mol2vec/unimol/hybrid. Para chemberta y
# molformer necesitamos pasarles la matriz pre-cargada como un override.
# Hacemos eso vía el mecanismo genérico de cache — guardamos los .npy primero,
# así cache.get los carga directamente.

def prebuild_npy_caches(cases):
    """Para cada caso y cada nueva representación, convierte el CSV de HF en
    .npy y lo guarda en output_representations/ para que cache.get lo use."""
    cache_dir = "output_representations"
    os.makedirs(cache_dir, exist_ok=True)
    for model_key in ("chemberta", "chemberta_77m"):
        getter = hf_getter(model_key)
        for case in cases:
            mat = getter(case)
            if mat is None:
                print(f"  SKIP {case.stem} / {model_key} — CSV no existe")
                continue
            npy_path = f"{cache_dir}/{case.stem}__{model_key}.npy"
            np.save(npy_path, mat)
            print(f"  cached {case.stem}__{model_key}.npy  ({mat.shape})")


def main():
    cases = build_cases()
    print(f"Cases: {len(cases)}")
    print()
    print("=" * 60)
    print("Paso 1/2: convertir CSVs HF a .npy cache")
    print("=" * 60)
    prebuild_npy_caches(cases)

    print()
    print("=" * 60)
    print("Paso 2/2: evaluar grid (solo chemberta + molformer)")
    print("=" * 60)

    t0 = time.time()
    df_new = run_grid(
        cases=cases,
        representations=["chemberta", "chemberta_77m"],
        cache_dir="output_representations",
        n_boot=1000,
        seed=42,
        verbose=True,
    )
    print(f"\nTier-3: {len(df_new):,} filas nuevas en {(time.time()-t0)/60:.1f} min")

    # Mergear con grid_results.csv existente
    grid_csv = f"{RESULTS_DIR}/grid_results.csv"
    old = pd.read_csv(grid_csv)
    merged = pd.concat([old, df_new], ignore_index=True)
    merged.to_csv(grid_csv, index=False)
    print(f"Grid mergeado: {len(merged):,} filas totales")

    # Regenerar summaries
    wr = win_rate_table(merged)
    wr.to_csv(f"{RESULTS_DIR}/win_rate_vs_ecfp4_tanimoto.csv", index=False)
    fam = summary_by_family(merged)
    fam.to_csv(f"{RESULTS_DIR}/summary_by_family.csv", index=False)
    final = (merged.groupby(["rep","metric"])["deviation_norm"]
             .mean().unstack("metric").round(3))
    final.to_csv(f"{RESULTS_DIR}/final_representation_x_metric.csv")

    print("\n--- TABLA FINAL (mean_norm por rep × metric) ---")
    print(final.to_string())


if __name__ == "__main__":
    main()
