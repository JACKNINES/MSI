"""
generate_hf_chemberta77m_only.py — corre SOLO ChemBERTa-77M-MTR
sobre los 15 datasets (reemplazo de MolFormer).
"""

import os, time
from hf_embeddings import generate_hf_vectors
from generate_hf_all import dataset_paths

os.makedirs("output_hf", exist_ok=True)

paths = dataset_paths()
print(f"Procesando {len(paths)} casos con ChemBERTa-77M-MTR\n")
t0 = time.time()
for path in paths:
    stem = os.path.splitext(os.path.basename(path))[0]
    n_mol = sum(1 for _ in open(path)) - 1
    print(f"\n--- {stem} ({n_mol:,} mol) ---")
    generate_hf_vectors(
        path=path,
        model_key="chemberta_77m",
        output_dir="output_hf",
        batch_size=64,
        use_cuda=True,
    )
print(f"\nChemBERTa-77M: terminado en {(time.time()-t0)/60:.1f} min")
