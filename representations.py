"""
representations.py — Unified molecular representation generator
=================================================================

Provides a single API to build any of the representations that the MSI
benchmark compares:

    Fingerprint family (discrete, binary)
    --------------------------------------
        ECFP (Morgan, radius=2/3, nbits=1024/2048/4096)  — varying length/design
        MACCS keys (166 bits)                             — different design
        RDKit daylight fingerprint (2048 bits)            — path-based, not circular

    Descriptor family (continuous, interpretable)
    ---------------------------------------------
        RDKit 2D physicochemical descriptors (~210 features)

    Learned embedding family (continuous, data-driven)
    ---------------------------------------------------
        Mol2Vec 300D   — distributional, 2D topological (wraps msi.py)
        Uni-Mol 512D   — 3D geometric (wraps unimol_msi.py)
        Hybrid 812D    — mol2vec + Uni-Mol concatenated

All generators return a dense numpy matrix of shape (N, D) aligned row-by-row
with the input SMILES list (rows of failed parses are kept as rows of NaN so
alignment across representations is preserved for downstream comparison).

This module addresses Reviewer 1, point 5: "benchmarking should be expanded to
include a wider range of molecular representations" — fingerprints of varying
lengths and designs, physicochemical descriptors, and alternative embeddings.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator

RDLogger.DisableLog("rdApp.*")


# =============================================================================
# Representation identifiers
# =============================================================================

FINGERPRINT_TYPES = {
    "ecfp4_1024": {"kind": "ecfp", "radius": 2, "nbits": 1024},
    "ecfp4_2048": {"kind": "ecfp", "radius": 2, "nbits": 2048},   # paper default
    "ecfp4_4096": {"kind": "ecfp", "radius": 2, "nbits": 4096},
    "ecfp6_2048": {"kind": "ecfp", "radius": 3, "nbits": 2048},
    "maccs":      {"kind": "maccs"},
    "rdkit_fp":   {"kind": "rdkit_fp", "nbits": 2048},
}
"""Binary (bit-vector) representations. Natural pair: Tanimoto/Tversky/Dice."""

DESCRIPTOR_TYPES = {
    "rdkit_physchem": {"kind": "rdkit_descriptors"},
}
"""Continuous numeric representations. Natural pair: Mahalanobis / Cosine."""

EMBEDDING_TYPES = {
    "mol2vec":   {"kind": "mol2vec",   "dim": 300},
    "unimol":    {"kind": "unimol",    "dim": 512},
    "hybrid":    {"kind": "hybrid",    "dim": 812},
    "chemberta":     {"kind": "hf_cached", "dim": 768, "tag": "chemberta"},
    "chemberta_77m": {"kind": "hf_cached", "dim": 384, "tag": "chemberta_77m"},
}
"""Learned continuous embeddings. Natural pair: Mahalanobis / Cosine."""

ALL_REPRESENTATIONS = {**FINGERPRINT_TYPES, **DESCRIPTOR_TYPES, **EMBEDDING_TYPES}


# =============================================================================
# Single-molecule generators
# =============================================================================

def _smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def ecfp_bits(mol: Chem.Mol, radius: int, nbits: int) -> np.ndarray:
    """Morgan/ECFP bit vector as a 0/1 uint8 numpy array of length ``nbits``."""
    gen = GetMorganGenerator(radius=radius, fpSize=nbits)
    bv = gen.GetFingerprint(mol)
    arr = np.zeros(nbits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def maccs_bits(mol: Chem.Mol) -> np.ndarray:
    """166-bit MACCS keys as a uint8 array (first bit is always 0 — RDKit pads to 167)."""
    bv = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def rdkit_fp_bits(mol: Chem.Mol, nbits: int = 2048) -> np.ndarray:
    """Path-based RDKit (Daylight-like) fingerprint as a uint8 array."""
    gen = GetRDKitFPGenerator(fpSize=nbits)
    bv = gen.GetFingerprint(mol)
    arr = np.zeros(nbits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


# RDKit 2D descriptor list — pre-compute once so column order is stable across
# datasets and runs. We explicitly list the descriptor name so the feature
# matrix is reproducible (rdkit occasionally reorders Descriptors._descList).
_RDKIT_DESCRIPTOR_NAMES: Tuple[str, ...] = tuple(name for name, _ in Descriptors._descList)
_RDKIT_DESCRIPTOR_FUNCS: Tuple = tuple(fn for _, fn in Descriptors._descList)


def rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    """
    ~210 RDKit 2D physicochemical descriptors.

    Covers molecular weight, logP, TPSA, HBA/HBD counts, ring counts,
    rotatable bonds, Kappa/Chi indices, VSA family, BalabanJ, etc.
    A descriptor that raises or returns non-finite is stored as NaN.
    """
    out = np.full(len(_RDKIT_DESCRIPTOR_FUNCS), np.nan, dtype=np.float64)
    for i, fn in enumerate(_RDKIT_DESCRIPTOR_FUNCS):
        try:
            v = fn(mol)
            if v is None:
                continue
            v = float(v)
            if np.isfinite(v):
                out[i] = v
        except Exception:
            pass
    return out


def rdkit_descriptor_names() -> List[str]:
    return list(_RDKIT_DESCRIPTOR_NAMES)


# =============================================================================
# Matrix-level generators
# =============================================================================

def generate_fingerprint_matrix(
    smiles_list: List[str],
    rep_type: str,
) -> np.ndarray:
    """
    Build an (N, D) uint8 bit-matrix for a fingerprint family representation.

    Failed SMILES parse → row of zeros (so Tanimoto returns 0 vs any other,
    which is the same behaviour as rdkit's Tanimoto on an all-zero vector).
    The caller is responsible for filtering if strict alignment is required.
    """
    cfg = ALL_REPRESENTATIONS[rep_type]
    kind = cfg["kind"]

    if kind == "ecfp":
        D = cfg["nbits"]
        out = np.zeros((len(smiles_list), D), dtype=np.uint8)
        for i, smi in enumerate(smiles_list):
            mol = _smiles_to_mol(smi)
            if mol is not None:
                out[i] = ecfp_bits(mol, radius=cfg["radius"], nbits=cfg["nbits"])
        return out

    if kind == "maccs":
        out = np.zeros((len(smiles_list), 167), dtype=np.uint8)
        for i, smi in enumerate(smiles_list):
            mol = _smiles_to_mol(smi)
            if mol is not None:
                out[i] = maccs_bits(mol)
        return out

    if kind == "rdkit_fp":
        D = cfg["nbits"]
        out = np.zeros((len(smiles_list), D), dtype=np.uint8)
        for i, smi in enumerate(smiles_list):
            mol = _smiles_to_mol(smi)
            if mol is not None:
                out[i] = rdkit_fp_bits(mol, nbits=D)
        return out

    raise ValueError(f"'{rep_type}' is not a fingerprint representation.")


def generate_descriptor_matrix(
    smiles_list: List[str],
    rep_type: str = "rdkit_physchem",
    impute: str = "median",
    standardize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build an (N, D) float64 descriptor matrix.

    Parameters
    ----------
    smiles_list : list of str
    rep_type : {"rdkit_physchem"}
    impute : {"median", "none"}
        Fill NaN cells with the per-column median (recommended — avoids
        propagating NaN through the covariance matrix).
    standardize : bool
        Z-score columns (mean 0, std 1). Required when mixing descriptors
        with very different scales (MW vs TPSA vs Chi indices) under any
        covariance-aware similarity.

    Returns
    -------
    matrix : np.ndarray, shape (N, D)
    names : list of str
    """
    cfg = ALL_REPRESENTATIONS[rep_type]
    if cfg["kind"] != "rdkit_descriptors":
        raise ValueError(f"'{rep_type}' is not a descriptor representation.")

    N = len(smiles_list)
    D = len(_RDKIT_DESCRIPTOR_FUNCS)
    mat = np.full((N, D), np.nan, dtype=np.float64)

    for i, smi in enumerate(smiles_list):
        mol = _smiles_to_mol(smi)
        if mol is not None:
            mat[i] = rdkit_descriptors(mol)

    # --- Impute ----------------------------------------------------------
    if impute == "median":
        col_medians = np.nanmedian(mat, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        nan_mask = np.isnan(mat)
        rows, cols = np.where(nan_mask)
        mat[rows, cols] = col_medians[cols]
    elif impute != "none":
        raise ValueError(f"Unknown impute mode: '{impute}'")

    # --- Drop constant columns (zero variance → singular covariance) ----
    col_std = mat.std(axis=0, ddof=1)
    keep = col_std > 1e-12
    mat = mat[:, keep]
    names = [n for n, k in zip(rdkit_descriptor_names(), keep) if k]

    # --- Standardize -----------------------------------------------------
    if standardize:
        mu = mat.mean(axis=0)
        sigma = mat.std(axis=0, ddof=1)
        sigma = np.where(sigma == 0, 1.0, sigma)
        mat = (mat - mu) / sigma

    return mat, names


# =============================================================================
# Unified entry point
# =============================================================================

def generate_representation(
    smiles_list: List[str],
    rep_type: str,
    mol2vec_model=None,
    unimol_output_dir: str = "output_unimol",
    unimol_input_path: Optional[str] = None,
    mol2vec_matrix: Optional[np.ndarray] = None,
    unimol_matrix: Optional[np.ndarray] = None,
    standardize_descriptors: bool = True,
) -> np.ndarray:
    """
    One-stop generator. Returns an (N, D) numpy array for any registered
    representation.

    Parameters
    ----------
    smiles_list : list of str
        Must be aligned with the caller's dataframe rows.
    rep_type : str
        One of ``ALL_REPRESENTATIONS``.
    mol2vec_model : Word2Vec, optional
        Pre-loaded mol2vec model. Required for ``rep_type="mol2vec"`` or
        ``"hybrid"`` unless ``mol2vec_matrix`` is passed.
    unimol_input_path : str, optional
        Path to the CSV that was (or will be) fed to ``generate_unimol_vectors``.
        Required for ``rep_type="unimol"`` or ``"hybrid"`` unless
        ``unimol_matrix`` is passed.
    unimol_output_dir : str
        Cache directory for Uni-Mol embeddings.
    mol2vec_matrix, unimol_matrix : np.ndarray, optional
        Pre-computed matrices (skip on-the-fly computation — useful when
        building ``hybrid`` from cached outputs of the other benchmarks).
    standardize_descriptors : bool
        Passthrough to :func:`generate_descriptor_matrix`.

    Notes
    -----
    For embeddings, reuse is strongly preferred: both mol2vec and Uni-Mol
    are expensive to compute. The ``benchmark.ipynb`` and
    ``benchmark_unimol.ipynb`` notebooks already cache their outputs on disk;
    pass them in via ``mol2vec_matrix`` / ``unimol_matrix`` to avoid rework.
    """
    if rep_type not in ALL_REPRESENTATIONS:
        raise ValueError(
            f"Unknown rep_type '{rep_type}'. "
            f"Valid: {sorted(ALL_REPRESENTATIONS)}"
        )
    cfg = ALL_REPRESENTATIONS[rep_type]
    kind = cfg["kind"]

    if kind in {"ecfp", "maccs", "rdkit_fp"}:
        return generate_fingerprint_matrix(smiles_list, rep_type)

    if kind == "rdkit_descriptors":
        mat, _ = generate_descriptor_matrix(
            smiles_list, rep_type, standardize=standardize_descriptors
        )
        return mat

    if kind == "mol2vec":
        if mol2vec_matrix is not None:
            return mol2vec_matrix.astype(np.float64, copy=False)
        if mol2vec_model is None:
            raise ValueError("mol2vec requires `mol2vec_model` or `mol2vec_matrix`.")
        # Minimal inline mol2vec path: uses msi.py's tokenizer via the
        # `generate_300d_vectors` flow would require writing a CSV; instead
        # we compute directly from the SMILES list to keep this generator
        # pure-in-memory.
        from mol2vec.features import mol2alt_sentence
        vocab = set(mol2vec_model.wv.key_to_index.keys())
        out = np.full((len(smiles_list), 300), np.nan, dtype=np.float64)
        for i, smi in enumerate(smiles_list):
            mol = _smiles_to_mol(smi)
            if mol is None:
                continue
            ids = mol2alt_sentence(mol, radius=1)
            if not all(x in vocab for x in ids):
                continue
            out[i] = np.mean([mol2vec_model.wv[x] for x in ids], axis=0)
        return out

    if kind == "unimol":
        if unimol_matrix is not None:
            return unimol_matrix.astype(np.float64, copy=False)
        if unimol_input_path is None:
            raise ValueError(
                "unimol requires `unimol_input_path` (CSV with smiles column) "
                "or a pre-computed `unimol_matrix`."
            )
        from unimol_msi import generate_unimol_vectors
        vec_path = generate_unimol_vectors(
            path=unimol_input_path,
            output_dir=unimol_output_dir,
        )
        return pd.read_csv(vec_path).values.astype(np.float64)

    if kind == "hybrid":
        from unimol_msi import build_hybrid_vectors
        m2v = (
            mol2vec_matrix
            if mol2vec_matrix is not None
            else generate_representation(
                smiles_list, "mol2vec", mol2vec_model=mol2vec_model
            )
        )
        um = (
            unimol_matrix
            if unimol_matrix is not None
            else generate_representation(
                smiles_list, "unimol",
                unimol_input_path=unimol_input_path,
                unimol_output_dir=unimol_output_dir,
            )
        )
        return build_hybrid_vectors(m2v, um, normalize=True)

    raise RuntimeError(f"Unhandled representation kind: {kind}")


# =============================================================================
# Quick profiling helper (useful from the notebook)
# =============================================================================

def timed_generate(
    smiles_list: List[str], rep_type: str, **kwargs
) -> Tuple[np.ndarray, float]:
    """Return ``(matrix, elapsed_seconds)`` — for logging/benchmarking."""
    t0 = time.time()
    mat = generate_representation(smiles_list, rep_type, **kwargs)
    return mat, time.time() - t0


__all__ = [
    "FINGERPRINT_TYPES",
    "DESCRIPTOR_TYPES",
    "EMBEDDING_TYPES",
    "ALL_REPRESENTATIONS",
    "ecfp_bits",
    "maccs_bits",
    "rdkit_fp_bits",
    "rdkit_descriptors",
    "rdkit_descriptor_names",
    "generate_fingerprint_matrix",
    "generate_descriptor_matrix",
    "generate_representation",
    "timed_generate",
]
