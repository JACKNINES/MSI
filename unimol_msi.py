"""
unimol_msi.py — MSI con embeddings 3D de Uni-Mol
=================================================

Extiende el pipeline MSI reemplazando mol2vec (2D, topologico) con
Uni-Mol (3D, conformacional). Tambien soporta embeddings hibridos
concatenando ambos espacios (812D).

Requiere: pip install unimol-tools

Funciones principales
---------------------
    generate_unimol_vectors  Extraer embeddings 512D de Uni-Mol
    compute_theta            Angulo de Mahalanobis generico (cualquier dim)
    cosine_similarity        Similitud coseno estandar
    build_hybrid_vectors     Concatenar mol2vec 300D + Uni-Mol 512D
    load_mol2vec_results     Cargar resultados existentes del benchmark mol2vec
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# ═══════════════════════════════════════════════════════════════════════
# Constantes
# ═══════════════════════════════════════════════════════════════════════

REGULARIZATION_LAMBDA: float = 1e-5
"""Regularizacion de Tikhonov — identico a msi.py."""

UNIMOL_DIM: int = 512
"""Dimensionalidad del CLS token de Uni-Mol v1."""

MOL2VEC_DIM: int = 300
"""Dimensionalidad de mol2vec (para referencia)."""

DEFAULT_BATCH_SIZE: int = 64
"""Batch size por defecto para inferencia de Uni-Mol."""


# ═══════════════════════════════════════════════════════════════════════
# Extraccion de embeddings Uni-Mol
# ═══════════════════════════════════════════════════════════════════════

def generate_unimol_vectors(
    path: str,
    output_dir: str = "output",
    output_name: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str = "unimolv1",
    use_cuda: bool = True,
    smiles_column: str = "smiles",
) -> str:
    """
    Extrae embeddings CLS de 512D usando Uni-Mol desde un CSV con SMILES.

    Los vectores se guardan en disco y se reutilizan si ya existen (cache).

    Parameters
    ----------
    path : str
        Ruta al CSV. Debe contener una columna con SMILES.
    output_dir : str
        Directorio donde guardar el CSV de vectores.
    output_name : str, optional
        Nombre del archivo de salida. Si None, se deriva de ``path``.
    batch_size : int
        Tamano de batch para inferencia.
    model_name : str
        ``'unimolv1'`` (209M conf.) o ``'unimolv2'`` (800M conf.).
    use_cuda : bool
        Usar GPU (CUDA) si esta disponible.
    smiles_column : str
        Nombre de la columna con SMILES en el CSV.

    Returns
    -------
    str
        Ruta al CSV guardado con 512 columnas numericas (una fila por molecula).
    """
    from unimol_tools import UniMolRepr

    os.makedirs(output_dir, exist_ok=True)

    # --- Leer SMILES ---
    df = pd.read_csv(path)
    if smiles_column not in df.columns:
        raise ValueError(
            f"El CSV no contiene la columna '{smiles_column}'. "
            f"Columnas encontradas: {list(df.columns)}"
        )
    smiles_list = df[smiles_column].tolist()
    n_total = len(smiles_list)

    # --- Determinar ruta de salida ---
    stem = os.path.splitext(os.path.basename(path))[0]
    if output_name is None:
        output_name = f"{stem}_unimol_512d_vectors.csv"
    output_path = os.path.join(output_dir, output_name)

    # --- Verificar cache ---
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        n_nan_rows = existing.isna().any(axis=1).sum()
        if len(existing) == n_total and n_nan_rows == 0:
            print(f"  [Cache] {output_path} ({n_total:,} moleculas)")
            return output_path
        if n_nan_rows > 0:
            print(f"  [Cache corrupto] {n_nan_rows:,}/{len(existing):,} filas con NaN. Regenerando...")
        else:
            print(f"  [Cache invalido] {len(existing):,} vs {n_total:,} filas. Regenerando...")

    # --- Detectar dispositivo ---
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    if use_cuda and not cuda_available:
        print("  NOTA: CUDA no disponible. Usando CPU (sera mas lento).")
        use_cuda = False

    # --- Inicializar modelo Uni-Mol ---
    print(f"  Inicializando Uni-Mol ({model_name})...")
    t_init = time.time()
    clf = UniMolRepr(
        data_type="molecule",
        remove_hs=False,
        batch_size=batch_size,
        model_name=model_name,
        use_cuda=use_cuda,
    )
    print(f"  Modelo listo en {time.time() - t_init:.1f}s")

    # --- Extraer embeddings por batches ---
    print(f"  Extrayendo {n_total:,} embeddings (batch_size={batch_size})...")
    t0 = time.time()

    all_vectors = []
    n_failed = 0
    n_batches = (n_total + batch_size - 1) // batch_size

    for i in range(0, n_total, batch_size):
        batch_smiles = smiles_list[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Progreso cada 100 batches, al inicio y al final
        if batch_num % 100 == 0 or batch_num == n_batches or batch_num == 1:
            elapsed = time.time() - t0
            done = i + len(batch_smiles)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_total - done) / rate if rate > 0 else 0
            print(
                f"    [{batch_num:>5d}/{n_batches}] "
                f"{done:>7,}/{n_total:,} mol  "
                f"({elapsed:>5.0f}s, ~{eta:.0f}s restantes)"
            )

        try:
            # get_repr con return_atomic_reprs=False retorna una LISTA de arrays (N, 512)
            # get_repr con return_atomic_reprs=True  retorna un DICT con 'cls_repr'
            repr_result = clf.get_repr(batch_smiles, return_atomic_reprs=False)
            if isinstance(repr_result, dict):
                vectors = np.array(repr_result["cls_repr"])
            elif isinstance(repr_result, list):
                vectors = np.vstack([np.array(v) for v in repr_result])  # list of (512,) → (batch, 512)
            else:
                vectors = np.array(repr_result)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            all_vectors.append(vectors)
        except Exception as e:
            print(f"    WARN batch {batch_num}: {e}")
            all_vectors.append(np.full((len(batch_smiles), UNIMOL_DIM), np.nan))
            n_failed += len(batch_smiles)

    all_vectors = np.vstack(all_vectors)  # (N, 512)
    elapsed = time.time() - t0

    # --- Guardar ---
    vec_df = pd.DataFrame(
        all_vectors, columns=[str(i) for i in range(all_vectors.shape[1])]
    )
    vec_df.to_csv(output_path, index=False)

    print(
        f"  Completado: {all_vectors.shape[0]:,} x {all_vectors.shape[1]}D "
        f"en {elapsed:.1f}s"
    )
    if n_failed > 0:
        print(f"  ADVERTENCIA: {n_failed:,} moleculas fallaron (NaN)")

    return output_path


# ═══════════════════════════════════════════════════════════════════════
# Calculo de theta (angulo de Mahalanobis) — generico
# ═══════════════════════════════════════════════════════════════════════

def compute_theta(
    vectors: np.ndarray,
    ref_idx: int = 0,
    lambda_reg: float = REGULARIZATION_LAMBDA,
    diagonal_only: bool = False,
    standardize: bool = True,
) -> np.ndarray:
    """
    Calcula el angulo de Mahalanobis (theta) en un espacio de embeddings
    de dimension arbitraria.

    Funciona identicamente para mol2vec 300D, Uni-Mol 512D, o hibrido 812D.
    La implementacion es identica a ``msi.py`` pero parametrizada.

    Parameters
    ----------
    vectors : np.ndarray
        Matriz (N, D) de embeddings.
    ref_idx : int
        Indice de la molecula de referencia (fila 0 = referencia).
    lambda_reg : float
        Regularizacion de Tikhonov sumada a la diagonal de la covarianza.
    diagonal_only : bool
        Si True, usa solo la diagonal (varianza, sin correlaciones cruzadas).
    standardize : bool
        Si True (default), estandariza cada dimension a media=0, std=1 antes
        de calcular la covarianza.  Esto equivale a usar la **matriz de
        correlacion** en lugar de la covarianza cruda, y estabiliza la
        inversion numerica para embeddings de alta varianza (e.g. Uni-Mol
        512D, varianza ~50x mayor que mol2vec).

    Returns
    -------
    np.ndarray
        Array (N,) de angulos theta en grados. 0 = identica a la referencia.
    """
    D = vectors.shape[1]

    if standardize:
        mu = vectors.mean(axis=0)
        sigma = vectors.std(axis=0, ddof=1)
        sigma = np.where(sigma == 0, 1.0, sigma)   # evitar div/0
        vectors = (vectors - mu) / sigma

    # Covarianza muestral (ddof=1, como pandas .cov())
    cov = np.cov(vectors, rowvar=False)  # (D, D)

    if diagonal_only:
        cov = np.diag(np.diag(cov))  # Solo varianza por dimension

    # Regularizacion de Tikhonov + inversion
    cov += np.eye(D) * lambda_reg
    inv_cov = np.linalg.inv(cov)

    # Vector de referencia
    ref_vec = vectors[ref_idx]

    # Angulo de Mahalanobis vectorizado
    #   cos(theta_i) = (v_i^T Sigma^-1 v_ref) / (||v_i||_M ||v_ref||_M)
    numerator = vectors @ inv_cov @ ref_vec  # (N,)

    vi_norm_sq = np.einsum("ij,ij->i", vectors, vectors @ inv_cov)  # (N,)
    vref_norm_sq = ref_vec @ inv_cov @ ref_vec  # escalar

    denominator = np.sqrt(np.maximum(vi_norm_sq, 0.0)) * np.sqrt(
        max(vref_norm_sq, 0.0)
    )

    safe_denom = np.where(denominator == 0, 1.0, denominator)
    cosine_vals = np.clip(numerator / safe_denom, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cosine_vals))
    theta_deg = np.where(denominator == 0, 0.0, theta_deg)

    return theta_deg


# ═══════════════════════════════════════════════════════════════════════
# Similitud coseno — generico
# ═══════════════════════════════════════════════════════════════════════

def cosine_similarity(vectors: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    """
    Similitud coseno estandar (sin transformacion de Mahalanobis).

    Parameters
    ----------
    vectors : np.ndarray
        Matriz (N, D).
    ref_idx : int
        Indice de la molecula de referencia.

    Returns
    -------
    np.ndarray
        Array (N,) de similitudes coseno en [-1, 1]. 1 = identica.
    """
    ref_vec = vectors[ref_idx]
    ref_norm = np.linalg.norm(ref_vec)
    vec_norms = np.linalg.norm(vectors, axis=1)

    safe_norms = np.where(vec_norms == 0, 1.0, vec_norms)
    safe_ref = max(ref_norm, 1e-12)

    return (vectors @ ref_vec) / (safe_norms * safe_ref)


# ═══════════════════════════════════════════════════════════════════════
# Embeddings hibridos (mol2vec + Uni-Mol)
# ═══════════════════════════════════════════════════════════════════════

def build_hybrid_vectors(
    mol2vec_vectors: np.ndarray,
    unimol_vectors: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Concatena mol2vec (300D) y Uni-Mol (512D) en un espacio hibrido (812D).

    Si ``normalize=True``, cada espacio se normaliza L2 por fila antes de
    concatenar, para que ninguno domine la covarianza.

    Parameters
    ----------
    mol2vec_vectors : np.ndarray
        Matriz (N, 300).
    unimol_vectors : np.ndarray
        Matriz (N, 512).
    normalize : bool
        Normalizar L2 por fila antes de concatenar.

    Returns
    -------
    np.ndarray
        Matriz (N, 812).
    """
    assert mol2vec_vectors.shape[0] == unimol_vectors.shape[0], (
        f"Numero de filas no coincide: "
        f"mol2vec={mol2vec_vectors.shape[0]} vs unimol={unimol_vectors.shape[0]}"
    )

    if normalize:
        m_norms = np.linalg.norm(mol2vec_vectors, axis=1, keepdims=True)
        u_norms = np.linalg.norm(unimol_vectors, axis=1, keepdims=True)
        m_norms = np.where(m_norms == 0, 1.0, m_norms)
        u_norms = np.where(u_norms == 0, 1.0, u_norms)
        mol2vec_vectors = mol2vec_vectors / m_norms
        unimol_vectors = unimol_vectors / u_norms

    return np.hstack([mol2vec_vectors, unimol_vectors])


# ═══════════════════════════════════════════════════════════════════════
# Utilidades para cargar resultados existentes del benchmark mol2vec
# ═══════════════════════════════════════════════════════════════════════

def get_mol2vec_paths(dataset_filename: str, output_dir: str = "output_benchmark") -> Dict[str, str]:
    """
    Deriva las rutas de los archivos generados por el pipeline mol2vec.

    Parameters
    ----------
    dataset_filename : str
        Ruta al CSV original del dataset (ej: ``data/benchmark/qm8_aniline.csv``).
    output_dir : str
        Directorio donde se guardaron los resultados.

    Returns
    -------
    dict
        Diccionario con claves ``'vectors_300d'``, ``'results'``, ``'vectors_2d'``.
    """
    stem = os.path.splitext(os.path.basename(dataset_filename))[0]
    return {
        "vectors_300d": os.path.join(output_dir, f"{stem}_300d_vectors.csv"),
        "vectors_2d": os.path.join(output_dir, f"{stem}_2d_vectors.csv"),
        "results": os.path.join(output_dir, f"{stem}_2d_vectors_against_it.csv"),
    }


def load_mol2vec_results(
    dataset_config: Dict[str, Any],
    output_dir: str = "output_benchmark",
) -> Optional[Dict[str, Any]]:
    """
    Carga los resultados existentes del pipeline mol2vec para un dataset.

    Parameters
    ----------
    dataset_config : dict
        Entrada de BENCHMARK_DATASETS (con ``'filename'``, ``'properties'``, etc.).
    output_dir : str
        Directorio con los resultados del benchmark original.

    Returns
    -------
    dict o None
        ``{'df': DataFrame, 'vectors_300d': ndarray, 'paths': dict}``
        o None si los archivos no existen.
    """
    paths = get_mol2vec_paths(dataset_config["filename"], output_dir)

    # Verificar que ambos archivos existen
    if not os.path.exists(paths["results"]):
        print(f"  WARN: No existe {paths['results']}")
        return None
    if not os.path.exists(paths["vectors_300d"]):
        print(f"  WARN: No existe {paths['vectors_300d']}")
        return None

    # Cargar
    df = pd.read_csv(paths["results"])
    vectors_300d = pd.read_csv(paths["vectors_300d"]).values.astype(np.float64)

    # Validar alineacion
    if len(df) != len(vectors_300d):
        print(
            f"  WARN: Desalineacion en {dataset_config['name']}: "
            f"results={len(df)}, vectors={len(vectors_300d)}"
        )
        return None

    return {
        "df": df,
        "vectors_300d": vectors_300d,
        "paths": paths,
    }
