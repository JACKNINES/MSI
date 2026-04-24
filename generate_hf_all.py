"""
generate_hf_all.py — corre ChemBERTa + MolFormer sobre los 15 datasets del benchmark.

Ejecuta en dirac:
    conda activate msi
    pip install transformers accelerate
    python3 generate_hf_all.py

Produce en output_hf/:
    <caso>_chemberta_768d_vectors.csv   (15 archivos, 768D cada uno)
    <caso>_molformer_768d_vectors.csv   (15 archivos)
"""

import os
import glob
import time

from hf_embeddings import generate_hf_vectors

BENCHMARK_DIR = "data/benchmark"
DATA_DIR      = "data"
OUTPUT_DIR    = "output_hf"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def dataset_paths():
    """Devuelve los 15 CSVs del benchmark (mismos que usa el grid)."""
    paths = sorted(glob.glob(f"{BENCHMARK_DIR}/qm8_*.csv")) + \
            sorted(glob.glob(f"{BENCHMARK_DIR}/qm9_*.csv")) + \
            sorted(glob.glob(f"{BENCHMARK_DIR}/esol_*.csv")) + \
            sorted(glob.glob(f"{BENCHMARK_DIR}/freesolv_*.csv")) + \
            sorted(glob.glob(f"{BENCHMARK_DIR}/lipo_*.csv"))
    # También incluir el QM9 aniline original si existe
    qm9_anilin = os.path.join(DATA_DIR, "qm9_anilin.csv")
    if os.path.exists(qm9_anilin) and qm9_anilin not in paths:
        paths.append(qm9_anilin)
    # Filtrar extras (solo mantener los 15 casos del grid)
    grid_cases = {
        "qm9_anilin", "qm9_pyridine",
        "qm8_aniline", "qm8_pyridine", "qm8_phenol",
        "esol_caffeine", "esol_naphthalene",
        "freesolv_toluene", "freesolv_ethanol",
        "lipo_diclofenac", "lipo_naproxen", "lipo_celecoxib",
        "lipo_simvastatin", "lipo_fluoxetine", "lipo_imipramine",
    }
    paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in grid_cases]
    return paths


def main():
    paths = dataset_paths()
    print(f"Procesando {len(paths)} casos del benchmark\n")

    for model_key in ("chemberta", "molformer"):
        print(f"\n{'='*70}")
        print(f"  MODELO: {model_key}")
        print(f"{'='*70}")
        t0 = time.time()

        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            n_mol = sum(1 for _ in open(path)) - 1
            print(f"\n--- {stem} ({n_mol:,} mol) ---")
            generate_hf_vectors(
                path=path,
                model_key=model_key,
                output_dir=OUTPUT_DIR,
                batch_size=64,
                use_cuda=True,
            )

        elapsed = time.time() - t0
        print(f"\n{model_key}: terminado en {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
