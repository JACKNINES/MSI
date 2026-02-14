"""
Mahalanobis Similarity Index (MSI) Module
==========================================

A unified framework for molecular similarity analysis using Mol2Vec embeddings,
Mahalanobis distance metrics, and hybrid clustering algorithms.

This module provides three core capabilities:

    1. **Molecular Vectorization** (Section 2):
       Convert SMILES strings into numerical vectors via Mol2Vec embeddings.
       Two representations are generated:
         - 2D vectors (PCA + t-SNE) for visualization
         - 300D vectors (raw Word2Vec means) for similarity computation

    2. **Similarity Analysis** (Sections 3–5):
       Compute multiple similarity metrics between a reference molecule and
       all others in the dataset:
         - Mahalanobis distance (d_M) and angle (theta_M)
         - Tanimoto similarity (TSI) via Morgan fingerprints
         - Cosine similarity in embedding space
         - Molecular weight

    3. **Clustering** (Section 7):
       Group molecules using K-Means and two hybrid variants (MK1, MK2)
       that combine Euclidean and Mahalanobis distances.

Authors: Roberto Bernal-Jaquez, Elliot Ridout-Buhl, 
Emiliano Montoya, León Alday-Toledo, Felipe Aparicio — UAM Cuajimalpa

Restructured: 2026

License: MIT
"""

# =============================================================================
# SECTION 0: IMPORTS AND CONSTANTS
# =============================================================================

import os
import time
import logging
import warnings
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
RDLogger.DisableLog("rdApp.*")  # Silence mol2vec's deprecated Morgan fingerprint warnings
from gensim.models import Word2Vec
from mol2vec.features import mol2alt_sentence
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from scipy.stats import describe
from scipy.spatial.distance import cosine as cosine_distance
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
tqdm.pandas()
sns.set_theme(style="white")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants — paper-specified parameters
# ---------------------------------------------------------------------------

REGULARIZATION_LAMBDA: float = 1e-5
"""Tikhonov regularizer added to the covariance diagonal (paper §Methodology, λ = 10⁻⁵)."""

MORGAN_RADIUS: int = 2
"""Radius for Morgan (ECFP) fingerprint generation (paper: ECFP4 → radius 2)."""

MORGAN_NBITS: int = 2048
"""Bit length for Morgan fingerprints (paper: 2048 bits)."""

MORGAN_GENERATOR = GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS)
"""Pre-built Morgan fingerprint generator (replaces deprecated GetMorganFingerprintAsBitVect)."""

MOL2VEC_RADIUS: int = 1
"""Radius for mol2alt_sentence substructure extraction."""

RANDOM_SEED: int = 42
"""Default random seed for reproducibility across the pipeline."""

TSNE_PERPLEXITY: int = 10
"""Perplexity for t-SNE dimensionality reduction."""

TSNE_ITERATIONS: int = 1000
"""Number of iterations for t-SNE optimization."""

PCA_COMPONENTS: int = 30
"""Number of PCA components before t-SNE (300D → 30D intermediate step)."""


# =============================================================================
# SECTION 1: PIPELINE METRICS
# =============================================================================

class PipelineMetrics:
    """
    Unified metrics tracker for the MSI pipeline.

    Collects and reports statistics from both the vectorization and
    similarity-analysis stages of the pipeline. All metrics are stored
    in the ``metrics`` dictionary.

    Tracked fields
    --------------
    invalid_smiles : int
        Number of SMILES strings that failed RDKit parsing.
    unknown_identifiers : int
        Number of molecules removed because at least one mol2vec
        identifier was absent from the Word2Vec vocabulary.
    covariance_condition_number : float or None
        Condition number of the 300D covariance matrix (``None`` if
        not yet computed).
    tanimoto_range : tuple of (float, float) or (None, None)
        (min, max) of Tanimoto similarities across the dataset.
    mahalanobis_stats : dict
        Descriptive statistics of Mahalanobis distances (nobs, mean,
        variance, min, max).
    runtime : dict of str → float
        Elapsed seconds for each pipeline stage.

    Examples
    --------
    >>> metrics = PipelineMetrics()
    >>> metrics.metrics["invalid_smiles"] = 42
    >>> metrics.log_summary()
    """

    def __init__(self) -> None:
        self.metrics: Dict[str, Any] = {
            "invalid_smiles": 0,
            "unknown_identifiers": 0,
            "covariance_condition_number": None,
            "tanimoto_range": (None, None),
            "mahalanobis_stats": {},
            "runtime": {},
        }

    def log_summary(self) -> None:
        """Log all collected metrics at INFO level."""
        logging.info("=== PIPELINE METRICS SUMMARY ===")
        for key, value in self.metrics.items():
            logging.info(f"  {key}: {value}")

    def __repr__(self) -> str:
        return f"PipelineMetrics({self.metrics})"


# =============================================================================
# SECTION 2: MOLECULAR VECTORIZATION
# =============================================================================

def _process_smiles(
    path: str,
    model: Word2Vec,
    metrics: PipelineMetrics,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load a SMILES CSV, parse molecules, extract mol2vec identifiers,
    and filter by model vocabulary.

    This is the shared pre-processing step for both 2D and 300D
    vectorization. Each SMILES string is:

    1. Parsed into an RDKit ``Mol`` object.
    2. Decomposed into mol2vec substructure identifiers (Morgan-based,
       radius = 1) via ``mol2alt_sentence``.
    3. Filtered: molecules with **any** identifier absent from the
       Word2Vec vocabulary are removed entirely.

    Parameters
    ----------
    path : str
        Path to a CSV file containing at minimum a ``smiles`` column.
        Files with a BOM (byte-order mark, common in PubChem exports)
        are handled automatically via ``encoding='utf-8-sig'``.
    model : model_300dim.pkl
        Pre-trained mol2vec Word2Vec model (300-dimensional).
    metrics : PipelineMetrics
        Metrics object — ``invalid_smiles`` and ``unknown_identifiers``
        fields are updated in place.

    Returns
    -------
    valid_df : pd.DataFrame
        Filtered DataFrame with added columns:

        - ``rdkit_mol``: RDKit Mol objects
        - ``identifiers``: list of mol2vec identifier strings per molecule
    unique_ids : list of str
        Deduplicated, insertion-ordered list of mol2vec identifiers
        found across all valid molecules.

    Raises
    ------
    ValueError
        If the input CSV does not contain a ``smiles`` column.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    initial_count = len(df)

    if "smiles" not in df.columns:
        raise ValueError(
            f"CSV at '{path}' must contain a 'smiles' column. "
            f"Found columns: {list(df.columns)}"
        )

    logging.info(f"Dataset size: {initial_count} molecules")

    # --- Step 1: Parse SMILES into RDKit Mol objects -------------------------
    df["rdkit_mol"] = df["smiles"].progress_apply(
        lambda x: Chem.MolFromSmiles(x) if pd.notnull(x) else None
    )

    valid_df = df.dropna(subset=["rdkit_mol"]).copy()
    metrics.metrics["invalid_smiles"] = initial_count - len(valid_df)

    # --- Step 2: Extract mol2vec identifiers ---------------------------------
    valid_df["identifiers"] = valid_df["rdkit_mol"].progress_apply(
        lambda mol: mol2alt_sentence(mol, radius=MOL2VEC_RADIUS)
    )

    # --- Step 3: Filter by vocabulary ----------------------------------------
    vocab = set(model.wv.key_to_index.keys())
    valid_df = valid_df[
        valid_df["identifiers"].apply(lambda ids: all(i in vocab for i in ids))
    ]
    metrics.metrics["unknown_identifiers"] = initial_count - len(valid_df)

    logging.info(
        f"After filtering: {len(valid_df)} molecules "
        f"(removed {metrics.metrics['invalid_smiles']} invalid SMILES, "
        f"{metrics.metrics['unknown_identifiers']} with unknown identifiers)"
    )

    # --- Step 4: Collect unique identifiers ----------------------------------
    all_ids = [uid for sublist in valid_df["identifiers"] for uid in sublist]
    unique_ids = list(OrderedDict.fromkeys(all_ids))

    return valid_df, unique_ids


def generate_2d_vectors(
    path: str,
    model: Word2Vec,
    output_dir: str = "output",
    preserve_columns: bool = False,
) -> str:
    """
    Generate 2D molecular embeddings via PCA + t-SNE dimensionality reduction.

    Pipeline
    --------
    1. Process SMILES via :func:`_process_smiles` (parse → identifiers → filter).
    2. Look up 300D Word2Vec vectors for each unique mol2vec identifier.
    3. Reduce identifier vectors: PCA(300 → 30) then t-SNE(30 → 2) with
       perplexity = 10, 1 000 iterations, cosine metric.
    4. For each molecule, aggregate its identifier 2D vectors using **np.sum**.
    5. Extract scalar components ``c1``, ``c2`` from the summed 2D vector.

    .. note:: **Why np.sum and not np.mean?**

       The sum aggregation is intentional. Molecules with more substructures
       produce larger-magnitude vectors, encoding structural complexity into
       the visualization space. The 300D vectors (used for actual similarity
       computation) use np.mean (see :func:`generate_300d_vectors`).
       Since Mahalanobis metrics normalize by covariance, the sum-based 2D
       representation is only used for visualization purposes.

    Parameters
    ----------
    path : str
        Path to input CSV with a ``smiles`` column (and optionally others).
    model : Word2Vec
        Pre-trained mol2vec model.
    output_dir : str, default ``"output"``
        Directory for the output CSV. Created if it does not exist.
    preserve_columns : bool, default ``False``
        If ``True``, all original CSV columns are preserved in the output.
        If ``False``, only ``smiles``, ``c1``, ``c2`` are saved.

    Returns
    -------
    str
        Absolute path to the generated CSV file:
        ``<output_dir>/<basename>_2d_vectors.csv``

    Output CSV schema
    -----------------
    - ``preserve_columns=False``: ``smiles, c1, c2``
    - ``preserve_columns=True``:  ``<all original columns>, c1, c2``

    Notes
    -----
    - The ``rdkit_mol`` and ``identifiers`` columns are always excluded
      from the output CSV (they do not serialize meaningfully).
    - A raw numpy ``vector`` column is NOT saved; only scalar ``c1``, ``c2``.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = PipelineMetrics()

    # --- Pre-processing ------------------------------------------------------
    df, unique_ids = _process_smiles(path, model, metrics)

    # --- Word2Vec embeddings for identifiers ---------------------------------
    identifier_vectors = np.array([model.wv[uid] for uid in unique_ids])

    # --- Dimensionality reduction: PCA(300→30) + t-SNE(30→2) ----------------
    pca = PCA(n_components=PCA_COMPONENTS)
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITERATIONS,
        metric="cosine",
        random_state=RANDOM_SEED,
    )
    reduced_vectors = tsne.fit_transform(pca.fit_transform(identifier_vectors))

    # --- Map identifiers to their 2D coordinates -----------------------------
    vector_map: Dict[str, np.ndarray] = {}
    for uid, vec in zip(unique_ids, reduced_vectors):
        vector_map[uid] = vec

    # --- Aggregate per-molecule (sum of identifier vectors) ------------------
    df["c1"] = df["identifiers"].apply(
        lambda ids: np.sum([vector_map[uid][0] for uid in ids])
    )
    df["c2"] = df["identifiers"].apply(
        lambda ids: np.sum([vector_map[uid][1] for uid in ids])
    )

    # --- Save ----------------------------------------------------------------
    basename = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(output_dir, f"{basename}_2d_vectors.csv")

    # Columns to exclude from CSV (non-serializable)
    exclude_cols = {"rdkit_mol", "identifiers"}

    if preserve_columns:
        cols_to_save = [c for c in df.columns if c not in exclude_cols]
    else:
        cols_to_save = ["smiles", "c1", "c2"]

    df[cols_to_save].to_csv(output_path, index=False)
    logging.info(f"2D vectors saved to {output_path}")

    metrics.log_summary()
    return output_path


def generate_300d_vectors(
    path: str,
    model: Word2Vec,
    output_dir: str = "output",
) -> str:
    """
    Generate 300-dimensional molecular embeddings by averaging mol2vec
    substructure vectors.

    Pipeline
    --------
    1. Process SMILES via :func:`_process_smiles`.
    2. For each molecule, compute the **mean** of its identifier Word2Vec
       vectors (``np.mean``, axis=0). This follows Eq. 1 of the paper:

       .. math::

           v_i = \\frac{1}{|N(i)|} \\sum_{j \\in N(i)} w_j

    3. Stack all molecule vectors into an (N × 300) matrix and save.

    Parameters
    ----------
    path : str
        Path to input CSV with a ``smiles`` column.
    model : Word2Vec
        Pre-trained mol2vec model.
    output_dir : str, default ``"output"``
        Directory for the output CSV.

    Returns
    -------
    str
        Absolute path to the generated CSV:
        ``<output_dir>/<basename>_300d_vectors.csv``

    Output CSV schema
    -----------------
    300 numeric columns (indices 0–299), one row per molecule.
    Row order matches the corresponding 2D vectors file exactly
    (same filtering is applied).

    Notes
    -----
    Uses ``np.mean`` (not ``np.sum``) for aggregation. Mean embedding is
    the standard practice for Word2Vec document representations in
    high-dimensional spaces and aligns with Eq. 1 of the paper.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics = PipelineMetrics()

    # --- Pre-processing ------------------------------------------------------
    df, unique_ids = _process_smiles(path, model, metrics)

    # --- Build identifier → vector lookup ------------------------------------
    identifier_vectors: Dict[str, np.ndarray] = {
        uid: model.wv[uid] for uid in unique_ids
    }

    # --- Aggregate per-molecule (mean of identifier vectors) -----------------
    df["vector"] = df["identifiers"].apply(
        lambda ids: np.mean([identifier_vectors[uid] for uid in ids], axis=0)
    )

    # --- Stack and save ------------------------------------------------------
    vectors = np.stack(df["vector"].values)
    basename = os.path.splitext(os.path.basename(path))[0]
    output_path = os.path.join(output_dir, f"{basename}_300d_vectors.csv")
    pd.DataFrame(vectors).to_csv(output_path, index=False)

    logging.info(f"300D vectors saved to {output_path}")
    return output_path


# =============================================================================
# SECTION 3: MAHALANOBIS DISTANCE AND ANGLE COMPUTATIONS
# =============================================================================

def mahalanobis_angle(
    vector_u: np.ndarray,
    vector_v: np.ndarray,
    inv_cov_matrix: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute the Mahalanobis-weighted angle between two vectors.

    The Mahalanobis angle generalizes the standard angle between vectors by
    weighting the inner product with the inverse covariance matrix Σ⁻¹,
    thereby accounting for correlations and variance differences across
    embedding dimensions.

    .. math::

        \\cos(\\theta_M) = \\frac{u^T \\Sigma^{-1} v}
        {\\sqrt{u^T \\Sigma^{-1} u} \\cdot \\sqrt{v^T \\Sigma^{-1} v}}

        \\theta_M = \\arccos\\bigl(\\cos(\\theta_M)\\bigr)

    Parameters
    ----------
    vector_u : np.ndarray, shape (d,)
        First vector (e.g., the reference molecule embedding).
    vector_v : np.ndarray, shape (d,)
        Second vector (e.g., a candidate molecule embedding).
    inv_cov_matrix : np.ndarray, shape (d, d)
        Inverse of the covariance matrix (Σ⁻¹).

    Returns
    -------
    angle_degrees : float
        Angle in degrees, range [0, 180].
    cosine_value : float
        Cosine of the angle, range [−1, 1].
    angle_radians : float
        Angle in radians, range [0, π].

    Notes
    -----
    Returns ``(0.0, 1.0, 0.0)`` if either vector has zero Mahalanobis norm
    (division-by-zero guard — this corresponds to the self-comparison case).
    """
    inner_product = vector_u.T @ inv_cov_matrix @ vector_v
    norm_u = np.sqrt(vector_u.T @ inv_cov_matrix @ vector_u)
    norm_v = np.sqrt(vector_v.T @ inv_cov_matrix @ vector_v)

    if norm_u == 0 or norm_v == 0:
        return 0.0, 1.0, 0.0

    cosine_value = np.clip(inner_product / (norm_u * norm_v), -1.0, 1.0)
    angle_radians = np.arccos(cosine_value)
    angle_degrees = np.degrees(angle_radians)

    return float(angle_degrees), float(cosine_value), float(angle_radians)


def mahalanobis_magnitude(
    vector: np.ndarray,
    mean_vector: np.ndarray,
    inv_cov_matrix: np.ndarray,
) -> float:
    """
    Compute the Mahalanobis distance from a vector to a reference point.

    .. math::

        \\|u - \\mu\\|_M = \\sqrt{(u - \\mu)^T \\, \\Sigma^{-1} \\, (u - \\mu)}

    Parameters
    ----------
    vector : np.ndarray, shape (d,)
        The vector to measure.
    mean_vector : np.ndarray, shape (d,)
        The centroid / mean / reference vector.
    inv_cov_matrix : np.ndarray, shape (d, d)
        Inverse covariance matrix (Σ⁻¹).

    Returns
    -------
    float
        Mahalanobis distance (non-negative scalar).
    """
    delta = vector - mean_vector
    return float(np.sqrt(delta.T @ inv_cov_matrix @ delta))


# =============================================================================
# SECTION 4: MOLECULAR PROPERTY COMPUTATIONS
# =============================================================================

def calculate_molecular_weight(smiles: str) -> Optional[float]:
    """
    Compute the molecular weight of a molecule from its SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule.

    Returns
    -------
    float or None
        Molecular weight in g/mol, or ``None`` if SMILES parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Descriptors.MolWt(mol)
    return None


def compute_tanimoto_similarity(smiles1: str, smiles2: str) -> Optional[float]:
    """
    Compute Tanimoto similarity between two molecules using Morgan fingerprints.

    Uses Morgan (circular) fingerprints with radius = 2 and 2 048 bits
    (ECFP4 configuration, as specified in the paper).

    .. math::

        T(A, B) = \\frac{|A \\cap B|}{|A| + |B| - |A \\cap B|}

    Parameters
    ----------
    smiles1 : str
        SMILES of the first molecule.
    smiles2 : str
        SMILES of the second molecule.

    Returns
    -------
    float or None
        Tanimoto similarity in [0, 1], or ``None`` if either SMILES is invalid.
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is not None and mol2 is not None:
        fp1 = MORGAN_GENERATOR.GetFingerprint(mol1)
        fp2 = MORGAN_GENERATOR.GetFingerprint(mol2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    return None


# =============================================================================
# SECTION 5: SIMILARITY ANALYSIS PIPELINE
# =============================================================================

def analyze_against_reference(
    path_2d: str,
    path_300d: str,
    output_dir: str = "output",
    reference_index: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the full MSI similarity analysis pipeline against a reference molecule.

    Computes all similarity metrics for every molecule in the dataset
    relative to a single reference molecule (by default, the first row).

    Pipeline stages
    ---------------
    1. **Data loading**: Read 2D vectors CSV and 300D embeddings CSV;
       validate that row counts match.
    2. **Molecular weight**: Parse SMILES → RDKit Mol → ``Descriptors.MolWt``.
    3. **Covariance matrix**: Compute sample covariance of 300D embeddings,
       regularize with λ = 10⁻⁵ (Tikhonov, paper-specified), invert.
    4. **Mahalanobis distance** (d_M): For each molecule vector v_i,

       .. math::
           d_M(v_i, v_{ref}) = \\sqrt{(v_i - v_{ref})^T \\, \\Sigma^{-1} \\, (v_i - v_{ref})}

    5. **Tanimoto similarity** (TSI): Morgan fingerprint (ECFP4) Tanimoto
       vs. reference.
    6. **Mahalanobis angle** (θ_M): Angle between each v_i and v_ref
       in Mahalanobis space.
    7. **Mahalanobis magnitude & projection**: Distance from the dataset
       mean and its cosine-weighted projection.
    8. **Embedding cosine similarity**: Standard cosine similarity in
       the original 300D space.
    9. **Save**: Write results to ``<output_dir>/<basename>_against_it.csv``.

    Parameters
    ----------
    path_2d : str
        Path to the 2D vectors CSV (must contain ``smiles``, ``c1``, ``c2``).
    path_300d : str
        Path to the 300D embeddings CSV (columns 0..299, numeric only).
    output_dir : str, default ``"output"``
        Directory for the output CSV file.
    reference_index : int, default ``0``
        Row index of the reference molecule (0-based).

    Returns
    -------
    result_df : pd.DataFrame
        DataFrame with all computed similarity columns (see below).
    metrics_dict : dict
        Pipeline metrics dictionary (timings, condition number, etc.).

    Output columns
    --------------
    =============================  ===========================================
    Column                         Description
    =============================  ===========================================
    smiles                         SMILES string
    c1, c2                         2D embedding coordinates
    weight                         Molecular weight (g/mol)
    mahalanobis_distance           d_M to reference (float)
    tanimoto_similarity            TSI vs reference (float, 0–1)
    theta                          θ_M in degrees (float — NOT string)
    cosine                         cos(θ_M) (float, −1 to 1)
    radians                        θ_M in radians (float)
    mahalanobis_magnitude          Distance from dataset mean (float)
    projection                     cosine × magnitude (float)
    embedding_similarity           Cosine similarity in 300D (float)
    =============================  ===========================================

    Plus any extra columns present in the 2D CSV (when ``preserve_columns``
    was ``True`` during vectorization).

    Fixes vs. original ``against_one()``
    -------------------------------------
    - ``theta`` remains as **float** (no f-string conversion to string).
    - Regularization uses **λ = 10⁻⁵** (not 10⁻⁸).
    - ``rdkit_mol`` column is **excluded** from the saved CSV.
    - Uses **vectorized** Mahalanobis computation via ``np.einsum``
      (≈10× faster than row-wise ``apply`` for large datasets).
    """
    start_time = time.time()
    metrics = PipelineMetrics()
    os.makedirs(output_dir, exist_ok=True)

    try:
        # ---- Stage 1: Data Loading & Validation -----------------------------
        load_start = time.time()
        primary_df = pd.read_csv(path_2d)
        vector_df = pd.read_csv(path_300d)

        if len(primary_df) != len(vector_df):
            raise ValueError(
                f"Row count mismatch: 2D file has {len(primary_df)} rows, "
                f"300D file has {len(vector_df)} rows. They must match."
            )

        vectors = vector_df.values  # shape: (N, 300)
        metrics.metrics["runtime"]["data_loading"] = time.time() - load_start

        # ---- Stage 2: Molecular Weight --------------------------------------
        weight_start = time.time()
        primary_df["rdkit_mol"] = primary_df["smiles"].apply(Chem.MolFromSmiles)

        weights = []
        invalid_count = 0
        for mol in primary_df["rdkit_mol"]:
            if mol is not None:
                weights.append(Descriptors.MolWt(mol))
            else:
                weights.append(np.nan)
                invalid_count += 1

        primary_df["weight"] = weights
        metrics.metrics["invalid_smiles"] = invalid_count
        metrics.metrics["runtime"]["weight_calculation"] = time.time() - weight_start

        # ---- Stage 3: Covariance Matrix (300D Embeddings) -------------------
        cov_start = time.time()
        covariance_matrix = vector_df.cov().values

        # Tikhonov regularization (paper: λ = 1e-5)
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * REGULARIZATION_LAMBDA

        cond_number = np.linalg.cond(covariance_matrix)
        metrics.metrics["covariance_condition_number"] = cond_number
        logging.info(f"Covariance matrix condition number: {cond_number:.2e}")

        inv_cov_matrix = np.linalg.inv(covariance_matrix)
        metrics.metrics["runtime"]["covariance_calculation"] = time.time() - cov_start

        # ---- Stage 4: Mahalanobis Distance to Reference (vectorized) --------
        mahalanobis_start = time.time()
        reference_vector = vectors[reference_index]

        # Vectorized computation: diff @ Σ⁻¹ @ diff^T per row via einsum
        diff = vectors - reference_vector  # (N, 300)
        left = diff @ inv_cov_matrix       # (N, 300)
        md_squared = np.einsum("ij,ij->i", diff, left)  # (N,)
        mahalanobis_distances = np.sqrt(np.maximum(md_squared, 0.0))

        primary_df["mahalanobis_distance"] = mahalanobis_distances

        valid_md = mahalanobis_distances[~np.isnan(mahalanobis_distances)]
        stats = describe(valid_md)
        metrics.metrics["mahalanobis_stats"] = {
            "nobs": int(stats.nobs),
            "mean": float(stats.mean),
            "variance": float(stats.variance),
            "min": float(stats.minmax[0]),
            "max": float(stats.minmax[1]),
        }
        metrics.metrics["runtime"]["mahalanobis_calculation"] = (
            time.time() - mahalanobis_start
        )

        # ---- Stage 5: Tanimoto Similarity (Morgan Fingerprints) -------------
        tanimoto_start = time.time()
        reference_mol = primary_df["rdkit_mol"].iloc[reference_index]
        if reference_mol is None:
            raise ValueError(
                f"Reference SMILES at index {reference_index} is invalid."
            )

        reference_fp = MORGAN_GENERATOR.GetFingerprint(reference_mol)

        def _tanimoto_for_mol(mol):
            if mol is None:
                return None
            try:
                fp = MORGAN_GENERATOR.GetFingerprint(mol)
                return DataStructs.TanimotoSimilarity(fp, reference_fp)
            except Exception as exc:
                logging.error(f"Tanimoto error: {exc}")
                return None

        primary_df["tanimoto_similarity"] = primary_df["rdkit_mol"].apply(
            _tanimoto_for_mol
        )

        valid_tanimoto = primary_df["tanimoto_similarity"].dropna()
        metrics.metrics["tanimoto_range"] = (
            float(valid_tanimoto.min()),
            float(valid_tanimoto.max()),
        )
        metrics.metrics["runtime"]["tanimoto_calculation"] = (
            time.time() - tanimoto_start
        )

        # ---- Stage 6: Mahalanobis Angle (vectorized) ------------------------
        angle_start = time.time()

        # Vectorized angle computation
        # numerator: v_i^T Σ⁻¹ v_ref  for each row i
        numerator = vectors @ inv_cov_matrix @ reference_vector  # (N,)

        # denominator: ||v_i||_M * ||v_ref||_M
        vi_norm_sq = np.einsum(
            "ij,ij->i", vectors, vectors @ inv_cov_matrix
        )  # (N,)
        vref_norm_sq = reference_vector @ inv_cov_matrix @ reference_vector
        denominator = np.sqrt(np.maximum(vi_norm_sq, 0.0)) * np.sqrt(
            max(vref_norm_sq, 0.0)
        )

        # Avoid division by zero
        safe_denom = np.where(denominator == 0, 1.0, denominator)
        cosine_values = np.clip(numerator / safe_denom, -1.0, 1.0)
        radians_values = np.arccos(cosine_values)
        theta_degrees = np.degrees(radians_values)

        # Handle zero-norm cases (self-comparison)
        theta_degrees = np.where(denominator == 0, 0.0, theta_degrees)
        cosine_values = np.where(denominator == 0, 1.0, cosine_values)
        radians_values = np.where(denominator == 0, 0.0, radians_values)

        primary_df["theta"] = theta_degrees       # float — NOT string
        primary_df["cosine"] = cosine_values
        primary_df["radians"] = radians_values
        metrics.metrics["runtime"]["angle_calculation"] = time.time() - angle_start

        # ---- Stage 7: Mahalanobis Magnitude & Projection --------------------
        mean_vector = vectors.mean(axis=0)
        diff_mean = vectors - mean_vector
        left_mean = diff_mean @ inv_cov_matrix
        mag_squared = np.einsum("ij,ij->i", diff_mean, left_mean)
        magnitudes = np.sqrt(np.maximum(mag_squared, 0.0))

        primary_df["mahalanobis_magnitude"] = magnitudes
        primary_df["projection"] = cosine_values * magnitudes

        # ---- Stage 8: Embedding Cosine Similarity (300D) --------------------
        # Vectorized cosine similarity: 1 - cosine_distance
        ref_norm = np.linalg.norm(reference_vector)
        vec_norms = np.linalg.norm(vectors, axis=1)
        safe_norms = np.where(vec_norms == 0, 1.0, vec_norms)
        safe_ref = max(ref_norm, 1e-12)
        dot_products = vectors @ reference_vector
        embedding_sim = dot_products / (safe_norms * safe_ref)

        primary_df["embedding_similarity"] = embedding_sim

        # ---- Stage 9: Save Results ------------------------------------------
        save_start = time.time()
        basename = os.path.splitext(os.path.basename(path_2d))[0]
        output_path = os.path.join(output_dir, f"{basename}_against_it.csv")

        # Drop non-serializable columns from DataFrame and CSV
        primary_df.drop(columns=["rdkit_mol"], inplace=True, errors="ignore")
        primary_df.to_csv(output_path, index=False)

        metrics.metrics["runtime"]["data_saving"] = time.time() - save_start
        logging.info(f"Results saved to {output_path}")

    except Exception:
        logging.error("Error in similarity analysis pipeline", exc_info=True)
        raise

    metrics.metrics["runtime"]["total"] = time.time() - start_time
    metrics.log_summary()

    return primary_df, metrics.metrics


# =============================================================================
# SECTION 6: MOLECULAR VISUALIZATION
# =============================================================================

def generate_molecule_map(
    df: pd.DataFrame,
    name: str,
    output_dir: str = "output",
    delta: float = 0.30,
    figsize: Tuple[int, int] = (60, 40),
    dpi: int = 300,
) -> str:
    """
    Create a 2D scatter plot of molecules with structural images overlaid.

    Each molecule is plotted at its (c1, c2) coordinates from the 2D
    vectorization. Overlapping positions are adjusted by random jittering
    to improve readability. RDKit molecular structure images are rendered
    at each point.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``smiles``, ``c1``, ``c2`` columns.
    name : str
        Dataset name for the plot title and output filename.
    output_dir : str, default ``"output"``
        Directory for the saved PNG image.
    delta : float, default ``0.30``
        Minimum distance threshold for overlap-based position adjustment.
        Points closer than ``delta`` to an existing point are jittered.
    figsize : tuple of (int, int), default ``(60, 40)``
        Figure size in inches.
    dpi : int, default ``300``
        Output resolution in dots per inch.

    Returns
    -------
    str
        Path to the saved PNG file:
        ``<output_dir>/map_of_molecules_<name>.png``

    Warnings
    --------
    For datasets with > 1 000 molecules, this function can be slow due to
    per-molecule image rendering. Consider subsampling first.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Creating molecule map for '{name}' ({len(df)} molecules)...")

    # --- Position adjustment to prevent overlaps -----------------------------
    positions = np.column_stack([df["c1"].values, df["c2"].values])
    adjusted = positions.copy()

    for i in range(len(adjusted)):
        xi, yi = adjusted[i]
        dists = np.linalg.norm(adjusted[:i] - np.array([xi, yi]), axis=1) if i > 0 else np.array([delta + 1])
        attempts = 0
        while np.any(dists < delta) and attempts < 100:
            xi += np.random.uniform(-delta, delta)
            yi += np.random.uniform(-delta, delta)
            dists = np.linalg.norm(adjusted[:i] - np.array([xi, yi]), axis=1) if i > 0 else np.array([delta + 1])
            attempts += 1
        adjusted[i] = [xi, yi]

    adj_c1, adj_c2 = adjusted[:, 0], adjusted[:, 1]

    # --- Create figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(adj_c1, adj_c2, c="blue", alpha=0.6, edgecolors="w", s=100)

    draw_opts = Draw.DrawingOptions()
    draw_opts.bgColor = (1, 1, 1, 0)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rendering molecules"):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            img = Draw.MolToImage(mol, size=(60, 60), options=draw_opts)
            imagebox = OffsetImage(img, zoom=0.6)
            pos_idx = df.index.get_loc(idx)
            ab = AnnotationBbox(
                imagebox,
                (adj_c1[pos_idx], adj_c2[pos_idx]),
                frameon=False,
            )
            ax.add_artist(ab)

    ax.set_xlabel("C1", fontsize=50)
    ax.set_ylabel("C2", fontsize=50)
    ax.set_title(f"Molecular Map — {name}", fontsize=50)
    ax.tick_params(labelsize=20)
    ax.set_xlim(adj_c1.min() - 50, adj_c1.max() + 50)
    ax.set_ylim(adj_c2.min() - 50, adj_c2.max() + 50)
    ax.grid(False)

    output_path = os.path.join(output_dir, f"map_of_molecules_{name}.png")
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Title": f"Vectorial Distribution of Molecules in {name}"},
    )
    plt.close(fig)

    logging.info(f"Molecule map saved to {output_path}")
    return output_path


# =============================================================================
# SECTION 7: CLUSTERING ALGORITHMS
# =============================================================================

# ---------------------------------------------------------------------------
# 7.1 Private helpers
# ---------------------------------------------------------------------------

def _initialize_centroids_mahalanobis(
    X: np.ndarray,
    n_clusters: int,
    init: str = "k-means++",
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Initialize cluster centroids using Mahalanobis-weighted k-means++.

    For ``k-means++`` initialization:

    1. Select the first centroid uniformly at random.
    2. For each subsequent centroid, compute the minimum Mahalanobis
       distance from every data point to the nearest existing centroid.
    3. Select the point with the **maximum** minimum-distance as the
       next centroid (ensures maximal spread in Mahalanobis space).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        StandardScaler-normalized data points.
    n_clusters : int
        Number of centroids to initialize.
    init : str, default ``"k-means++"``
        Initialization method: ``"k-means++"`` or ``"random"``.
    n_jobs : int, default ``-1``
        Number of parallel jobs for distance computation (``-1`` = all cores).

    Returns
    -------
    np.ndarray, shape (n_clusters, n_features)
        Initialized centroids.

    Raises
    ------
    ValueError
        If ``init`` is not ``"random"`` or ``"k-means++"``.
    """
    n_samples, n_features = X.shape
    centroids = np.zeros((n_clusters, n_features))

    if init == "random":
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        for i in range(n_clusters):
            centroids[i] = np.random.uniform(min_vals, max_vals)

    elif init == "k-means++":
        centroids[0] = X[np.random.randint(0, n_samples)]
        inv_covmat = np.linalg.inv(np.cov(X.T))

        for i in range(1, n_clusters):
            distances = Parallel(n_jobs=n_jobs)(
                delayed(
                    lambda x: min(
                        scipy_mahalanobis(x, c, inv_covmat)
                        for c in centroids[:i]
                    )
                )(x)
                for x in X
            )
            centroids[i] = X[np.argmax(distances)]
    else:
        raise ValueError(
            f"Unsupported initialization method: '{init}'. "
            f"Use 'random' or 'k-means++'."
        )

    return centroids


def _compute_mahalanobis_distance_matrix(
    X: np.ndarray,
    centroids: np.ndarray,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute pairwise Mahalanobis distances between data points and centroids.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data points.
    centroids : np.ndarray, shape (n_clusters, n_features)
        Cluster centroids.
    n_jobs : int, default ``-1``
        Parallelism for ``joblib.Parallel``.

    Returns
    -------
    np.ndarray, shape (n_samples, n_clusters)
        Distance matrix where entry ``[i, j]`` is the Mahalanobis distance
        from ``X[i]`` to ``centroids[j]``.
    """
    inv_covmat = np.linalg.inv(np.cov(X.T))

    def _distances_to_centroids(x):
        return [scipy_mahalanobis(x, c, inv_covmat) for c in centroids]

    result = Parallel(n_jobs=n_jobs)(
        delayed(_distances_to_centroids)(x) for x in X
    )
    return np.array(result)


def _compute_pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Euclidean distance matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data points.

    Returns
    -------
    np.ndarray, shape (n_samples, n_samples)
        Symmetric distance matrix.
    """
    return cdist(X, X, metric="euclidean")


# ---------------------------------------------------------------------------
# 7.2 Public clustering functions
# ---------------------------------------------------------------------------

def cluster_kmeans(
    path_2d: str,
    path_300d: str,
    n_clusters: int,
    output_dir: str = "output",
    random_state: int = 1,
) -> str:
    """
    Standard K-Means clustering on 300D molecular embeddings.

    Pipeline:

    1. Load 2D vectors CSV (metadata) and 300D embeddings.
    2. StandardScaler normalization on 300D vectors.
    3. K-Means (k-means++, Lloyd algorithm, 10 initializations).
    4. Append ``cluster`` column to 2D DataFrame.
    5. Save labeled CSV and cluster visualization PNG.

    Parameters
    ----------
    path_2d : str
        Path to 2D vectors CSV (must have ``smiles``, ``c1``, ``c2``).
    path_300d : str
        Path to 300D embeddings CSV (columns 0–299, numeric).
    n_clusters : int
        Number of clusters to generate.
    output_dir : str, default ``"output"``
        Output directory for labeled CSV and visualization.
    random_state : int, default ``1``
        Random seed for K-Means reproducibility.

    Returns
    -------
    str
        Path to the labeled output CSV:
        ``<output_dir>/<basename>_labeled_kmeans.csv``
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(path_2d)
    X = pd.read_csv(path_300d).values

    logging.info(f"K-Means clustering: {n_clusters} clusters on {X.shape[0]} molecules")

    # Normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=random_state,
        algorithm="lloyd",
    )
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Save labeled CSV
    basename = os.path.splitext(os.path.basename(path_2d))[0]
    output_path = os.path.join(output_dir, f"{basename}_labeled_kmeans.csv")
    df.to_csv(output_path, index=False)
    logging.info(f"Labeled dataset saved to {output_path}")

    # Visualization
    if "c1" in df.columns and "c2" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            df["c1"], df["c2"], c=df["cluster"], cmap="turbo", alpha=0.7
        )
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        ax.set_title(f"K-Means Clustering ({n_clusters} clusters)")
        plt.colorbar(scatter, ax=ax, label="Cluster")

        img_path = os.path.join(output_dir, f"{basename}_kmeans_clusters.png")
        fig.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Cluster plot saved to {img_path}")

    return output_path


def cluster_mk1(
    path_2d: str,
    path_300d: str,
    n_clusters: int,
    output_dir: str = "output",
    max_iter: int = 100,
    tol: float = 1e-4,
    init: str = "k-means++",
    n_jobs: int = -1,
) -> str:
    """
    MK1: Hybrid Euclidean + Mahalanobis K-Means clustering.

    Adapted from the thesis of Joshua Nelson. This algorithm alternates
    between Euclidean and Mahalanobis distance metrics within each
    iteration:

    1. Assign clusters using **Euclidean** distance.
    2. Recompute centroids as cluster means.
    3. Reassign clusters using **Mahalanobis** distance.
    4. Recompute centroids again.
    5. Check convergence (centroid movement < ``tol``).

    Centroids are initialized using Mahalanobis-weighted k-means++.

    Parameters
    ----------
    path_2d : str
        Path to 2D vectors CSV.
    path_300d : str
        Path to 300D embeddings CSV.
    n_clusters : int
        Number of clusters.
    output_dir : str, default ``"output"``
        Output directory for labeled CSV and centroids CSV.
    max_iter : int, default ``100``
        Maximum iterations before stopping.
    tol : float, default ``1e-4``
        Convergence tolerance on centroid movement (L2 norm).
    init : str, default ``"k-means++"``
        Centroid initialization method.
    n_jobs : int, default ``-1``
        Parallel jobs for Mahalanobis distance computation.

    Returns
    -------
    str
        Path to the labeled output CSV:
        ``<output_dir>/<basename>_mk1_labeled.csv``

    Notes
    -----
    **Bug fix**: CSV saving is performed **after** convergence (or reaching
    ``max_iter``), not inside the iteration loop as in the original code.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(path_2d)
    X_raw = pd.read_csv(path_300d).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    logging.info(f"MK1 clustering: {n_clusters} clusters, max_iter={max_iter}")

    # Initialize centroids
    centroids = _initialize_centroids_mahalanobis(X, n_clusters, init=init, n_jobs=n_jobs)
    labels = np.zeros(X.shape[0], dtype=int)

    for iteration in range(max_iter):
        old_centroids = centroids.copy()

        # Phase A: Euclidean assignment + centroid update
        labels = pairwise_distances_argmin(X, centroids)
        for j in range(n_clusters):
            mask = labels == j
            if np.any(mask):
                centroids[j] = X[mask].mean(axis=0)

        # Phase B: Mahalanobis assignment + centroid update
        distances = _compute_mahalanobis_distance_matrix(X, centroids, n_jobs=n_jobs)
        labels = np.argmin(distances, axis=1)
        for j in range(n_clusters):
            mask = labels == j
            if np.any(mask):
                centroids[j] = X[mask].mean(axis=0)

        # Convergence check
        movement = np.linalg.norm(centroids - old_centroids, axis=1)
        if np.all(movement < tol):
            logging.info(f"MK1 converged after {iteration + 1} iterations.")
            break
    else:
        logging.warning(f"MK1 reached max_iter ({max_iter}) without converging.")

    # --- Save results (OUTSIDE the loop — bug fix) ---------------------------
    basename = os.path.splitext(os.path.basename(path_2d))[0]

    df["mk1_cluster"] = labels
    output_path = os.path.join(output_dir, f"{basename}_mk1_labeled.csv")
    df.to_csv(output_path, index=False)

    centroids_df = pd.DataFrame(centroids)
    centroids_path = os.path.join(output_dir, f"centroids_{basename}_mk1.csv")
    centroids_df.to_csv(centroids_path, index=False)

    logging.info(f"MK1 labels saved to {output_path}")
    logging.info(f"MK1 centroids saved to {centroids_path}")

    return output_path


def cluster_mk2(
    path_2d: str,
    path_300d: str,
    n_clusters: int,
    w: int = 60,
    output_dir: str = "output",
    max_iter: int = 100,
    tol: float = 1e-4,
    n_jobs: int = -1,
) -> str:
    """
    MK2: W-parameter hybrid Euclidean + Mahalanobis K-Means clustering.

    Similar to MK1 but uses a different centroid initialization strategy
    based on a centrality parameter ``w``:

    1. Compute the full pairwise Euclidean distance matrix.
    2. Sort data points by their sum of distances to all others
       (centrality score: lower sum → more central).
    3. Select initial centroids by random sampling from the top-``w``
       most central points, removing each selected point from the pool.
    4. Iterate with the same Euclidean → Mahalanobis alternation as MK1.

    The ``w`` parameter controls how "central" the initial centroids are:

    - **Small w**: centroids start near the dense core of the dataset.
    - **Large w**: centroids can start further from center, increasing
      coverage.

    Parameters
    ----------
    path_2d : str
        Path to 2D vectors CSV.
    path_300d : str
        Path to 300D embeddings CSV.
    n_clusters : int
        Number of clusters.
    w : int, default ``60``
        Size of the candidate pool for initial centroid selection.
    output_dir : str, default ``"output"``
        Output directory.
    max_iter : int, default ``100``
        Maximum iterations.
    tol : float, default ``1e-4``
        Convergence tolerance.
    n_jobs : int, default ``-1``
        Parallel jobs for Mahalanobis computation.

    Returns
    -------
    str
        Path to the labeled output CSV:
        ``<output_dir>/<basename>_mk2_labeled.csv``
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(path_2d)
    X_raw = pd.read_csv(path_300d).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    logging.info(f"MK2 clustering: {n_clusters} clusters, w={w}, max_iter={max_iter}")

    n_samples, n_features = X.shape

    # --- Centrality-based initialization -------------------------------------
    euc_distances = _compute_pairwise_euclidean(X)
    sorted_indices = np.argsort(euc_distances.sum(axis=1))

    centroids = np.zeros((n_clusters, n_features))
    selected_indices: List[int] = []

    for i in range(n_clusters):
        candidate_pool = [idx for idx in sorted_indices[:w] if idx not in selected_indices]
        if len(candidate_pool) == 0:
            candidate_pool = [idx for idx in sorted_indices if idx not in selected_indices]
        chosen = np.random.choice(candidate_pool)
        centroids[i] = X[chosen]
        selected_indices.append(chosen)

    # --- Iterative clustering ------------------------------------------------
    labels = np.zeros(n_samples, dtype=int)

    for iteration in range(max_iter):
        old_centroids = centroids.copy()

        # Phase A: Euclidean assignment + centroid update
        labels = pairwise_distances_argmin(X, centroids)
        for j in range(n_clusters):
            mask = labels == j
            if np.any(mask):
                centroids[j] = X[mask].mean(axis=0)

        # Phase B: Mahalanobis assignment + centroid update
        distances = _compute_mahalanobis_distance_matrix(X, centroids, n_jobs=n_jobs)
        labels = np.argmin(distances, axis=1)
        for j in range(n_clusters):
            mask = labels == j
            if np.any(mask):
                centroids[j] = X[mask].mean(axis=0)

        # Convergence check
        movement = np.linalg.norm(centroids - old_centroids, axis=1)
        if np.all(movement < tol):
            logging.info(f"MK2 converged after {iteration + 1} iterations.")
            break
    else:
        logging.warning(f"MK2 reached max_iter ({max_iter}) without converging.")

    # --- Save results --------------------------------------------------------
    basename = os.path.splitext(os.path.basename(path_2d))[0]

    df["mk2_cluster"] = labels
    output_path = os.path.join(output_dir, f"{basename}_mk2_labeled.csv")
    df.to_csv(output_path, index=False)

    centroids_df = pd.DataFrame(centroids)
    centroids_path = os.path.join(output_dir, f"centroids_{basename}_mk2.csv")
    centroids_df.to_csv(centroids_path, index=False)

    logging.info(f"MK2 labels saved to {output_path}")
    logging.info(f"MK2 centroids saved to {centroids_path}")

    return output_path
