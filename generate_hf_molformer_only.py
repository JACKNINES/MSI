"""
generate_hf_molformer_only.py — corre SOLO MolFormer sobre los 15 datasets.

Usar cuando ChemBERTa ya terminó y solo falta este modelo.
"""

import os
import time

from hf_embeddings import generate_hf_vectors
from generate_hf_all import dataset_paths

OUTPUT_DIR = "output_hf"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    paths = dataset_paths()
    print(f"Procesando {len(paths)} casos con MolFormer\n")

    t0 = time.time()
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        n_mol = sum(1 for _ in open(path)) - 1
        print(f"\n--- {stem} ({n_mol:,} mol) ---")
        generate_hf_vectors(
            path=path,
            model_key="molformer",
            output_dir=OUTPUT_DIR,
            batch_size=64,
            use_cuda=True,
        )

    elapsed = time.time() - t0
    print(f"\nMolFormer: terminado en {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
