"""
hf_embeddings.py — wrappers para embeddings de HuggingFace (ChemBERTa, MolFormer)
=================================================================================

Análogo a unimol_msi.py pero usando el stack HuggingFace. Produce vectores
continuos que luego entran al grid como dos representaciones más:

    chemberta (768D) — Transformer bidireccional sobre SMILES
                        (seyonec/ChemBERTa-zinc-base-v1)
    molformer (768D) — Transformer de IBM sobre SMILES
                        (ibm/MoLFormer-XL-both-10pct)

Ambos son "semánticos distribucionales" como Mol2Vec pero con arquitectura
transformer, cubriendo el paradigma que nos faltaba en el benchmark.

Requiere: pip install transformers accelerate
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd


MODELS = {
    "chemberta": {
        "repo": "seyonec/ChemBERTa-zinc-base-v1",
        "dim": 768,
        "trust_remote_code": False,
    },
    # ChemBERTa-77M-MTR: 77M params (13× más grande que el anterior),
    # entrenado con MLM + multi-task regression sobre PubChem 10M.
    # Usa la API estándar de transformers — sin problemas de compatibilidad.
    # (MolFormer fue descartado por incompatibilidad con transformers >=4.40.)
    "chemberta_77m": {
        "repo": "DeepChem/ChemBERTa-77M-MTR",
        "dim": 384,
        "trust_remote_code": False,
    },
}


def generate_hf_vectors(
    path: str,
    model_key: str,
    output_dir: str = "output_hf",
    output_name: Optional[str] = None,
    batch_size: int = 64,
    use_cuda: bool = True,
    smiles_column: str = "smiles",
) -> str:
    """
    Extrae embeddings (CLS token) para un modelo HuggingFace sobre SMILES.

    Parameters
    ----------
    path : ruta al CSV con columna 'smiles'.
    model_key : "chemberta" o "molformer".
    output_dir : directorio de salida.
    output_name : nombre del CSV de salida (se deriva de path si es None).
    batch_size : tamaño de batch para inferencia.
    use_cuda : True para GPU, False para CPU.

    Returns
    -------
    Ruta al CSV con los vectores (N × dim).
    """
    if model_key not in MODELS:
        raise ValueError(f"model_key debe ser uno de {list(MODELS)}")
    cfg = MODELS[model_key]

    os.makedirs(output_dir, exist_ok=True)

    # --- Cargar SMILES ---
    df = pd.read_csv(path)
    if smiles_column not in df.columns:
        raise ValueError(f"CSV sin columna '{smiles_column}'. Encontradas: {list(df.columns)}")
    smiles_list = df[smiles_column].astype(str).tolist()
    n_total = len(smiles_list)

    # --- Cache (revisar ANTES de importar torch/transformers) ---
    stem = os.path.splitext(os.path.basename(path))[0]
    if output_name is None:
        output_name = f"{stem}_{model_key}_{cfg['dim']}d_vectors.csv"
    output_path = os.path.join(output_dir, output_name)

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if len(existing) == n_total and not existing.isna().any().any():
            print(f"  [cache] {output_path} ({n_total:,} mol)")
            return output_path
        print(f"  [cache inválido] regenerando...")

    # --- Imports pesados solo cuando se necesita inferencia ---
    import torch
    # Monkey-patch para compatibilidad con MolFormer (transformers.onnx fue removido en >=4.40)
    import sys, types
    if "transformers.onnx" not in sys.modules:
        _stub = types.ModuleType("transformers.onnx")
        _stub.OnnxConfig = type("OnnxConfig", (object,), {})
        sys.modules["transformers.onnx"] = _stub
    from transformers import AutoTokenizer, AutoModel

    # --- Device ---
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    if use_cuda and device == "cpu":
        print("  NOTA: CUDA no disponible, usando CPU (más lento)")

    # --- Modelo ---
    print(f"  Cargando {cfg['repo']}...")
    t_init = time.time()
    kwargs = {"trust_remote_code": True} if cfg["trust_remote_code"] else {}
    tokenizer = AutoTokenizer.from_pretrained(cfg["repo"], **kwargs)
    model = AutoModel.from_pretrained(cfg["repo"], **kwargs).to(device).eval()
    print(f"  Modelo listo en {time.time()-t_init:.1f}s (device={device})")

    # --- Inferencia por batches ---
    print(f"  Extrayendo {n_total:,} embeddings (batch_size={batch_size})...")
    t0 = time.time()
    all_vectors = []
    n_failed = 0
    n_batches = (n_total + batch_size - 1) // batch_size

    for i in range(0, n_total, batch_size):
        batch = smiles_list[i:i + batch_size]
        batch_num = i // batch_size + 1

        if batch_num % 100 == 0 or batch_num == n_batches or batch_num == 1:
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / rate if rate > 0 else 0
            print(f"    [{batch_num:>5d}/{n_batches}] "
                  f"{done:>7,}/{n_total:,} mol "
                  f"({elapsed:>5.0f}s, ~{eta:.0f}s restantes)")

        try:
            inputs = tokenizer(batch, padding=True, truncation=True,
                               return_tensors="pt", max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            # CLS token (primer token) — usado por ambos modelos como vector de molécula
            cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_vectors.append(cls_vecs.astype(np.float32))
        except Exception as e:
            print(f"    WARN batch {batch_num}: {e}")
            all_vectors.append(np.full((len(batch), cfg["dim"]), np.nan, dtype=np.float32))
            n_failed += len(batch)

    all_vectors = np.vstack(all_vectors)
    elapsed = time.time() - t0

    # --- Guardar ---
    vec_df = pd.DataFrame(all_vectors, columns=[str(j) for j in range(cfg["dim"])])
    vec_df.to_csv(output_path, index=False)
    print(f"  Completado: {all_vectors.shape[0]:,} × {all_vectors.shape[1]}D "
          f"en {elapsed:.1f}s")
    if n_failed > 0:
        print(f"  ADVERTENCIA: {n_failed:,} moléculas fallaron (NaN)")

    # Liberar memoria GPU
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python3 hf_embeddings.py <model_key> <csv_path>")
        print("model_key: chemberta | molformer")
        sys.exit(1)
    generate_hf_vectors(path=sys.argv[2], model_key=sys.argv[1])
