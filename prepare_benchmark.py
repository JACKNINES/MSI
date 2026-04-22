"""
Benchmark Data Preparation for MSI
===================================
Downloads and prepares ground-truth property datasets for the MSI definitive benchmark.
Creates one CSV per (dataset, reference_molecule) pair, with the reference in row 0.

Output: data/benchmark/<dataset>_<reference>.csv

Usage:
    python prepare_benchmark.py              # Full preparation (download + format)
    python prepare_benchmark.py --scan-only  # Only scan datasets for known molecules
"""

import os
import sys
import ssl
import subprocess
import urllib.request
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DIR = os.path.join("data", "benchmark", "raw")
BENCHMARK_DIR = os.path.join("data", "benchmark")

URLS = {
    "esol": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
    "freesolv": "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt",
    "lipophilicity": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "qm8": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv",
}

# Well-known molecules to scan for in each dataset (name -> SMILES)
#
# Grouped by size: small / medium / large. The large group is the one
# Reviewer 1 (point 4) is asking about: "larger and more structurally
# complex molecules". Any that are found in Lipophilicity / ESOL are used
# as reference molecules in addition to the original aniline + aspirin.
KNOWN_MOLECULES = {
    # --- small (MW < 150) -------------------------------------------------
    "methanol":       "CO",
    "ethanol":        "CCO",
    "acetic_acid":    "CC(=O)O",
    "acetone":        "CC(=O)C",
    "ethyl_acetate":  "CCOC(=O)C",
    "chloroform":     "ClC(Cl)Cl",
    "benzene":        "c1ccccc1",
    "toluene":        "Cc1ccccc1",
    "aniline":        "Nc1ccccc1",
    "pyridine":       "c1ccncc1",
    "phenol":         "Oc1ccccc1",
    "naphthalene":    "c1ccc2ccccc2c1",

    # --- medium (150 ≤ MW < 350) -----------------------------------------
    "aspirin":        "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen":      "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "caffeine":       "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "acetaminophen":  "CC(=O)Nc1ccc(O)cc1",
    "naproxen":       "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "diclofenac":     "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "metoprolol":     "COCCc1ccc(cc1)OCC(O)CNC(C)C",
    "glucose":        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "fluoxetine":     "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "imipramine":     "CN(C)CCCN1c2ccccc2CCc2ccccc21",
    "testosterone":   "CC12CCC3C(CCC4=CC(=O)CC[C@]34C)C1CC[C@@H]2O",

    # --- large (MW ≥ 350) — the "complex" references Reviewer 1 requested -
    "curcumin":       "O=C(/C=C/c1ccc(O)c(OC)c1)CC(=O)/C=C/c1ccc(O)c(OC)c1",  # 368, 2 rings + cross
    "celecoxib":      "Cc1ccc(cc1)-c1cc(C(F)(F)F)nn1-c1ccc(cc1)S(N)(=O)=O",   # 381, 4 rings + CF3
    "sildenafil":     "CCCc1nn(C)c2c1NC(=NC2=O)c1cc(ccc1OCC)S(=O)(=O)N1CCN(C)CC1",  # 474
    "losartan":       "CCCCc1nc(Cl)c(CO)n1Cc1ccc(cc1)-c1ccccc1-c1nnn[nH]1",   # 422
    "atorvastatin":   "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",  # 558
    "amoxicillin":    "C[C@@H]1(C(=O)O)N2[C@@H](CC2=O)SC(C)(C)1",  # simplified core
    "simvastatin":    "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12",
    "tamoxifen":      "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",          # 371
}

# Convenience: mark which tier every molecule belongs to (used to decide
# which ones serve as references for point-4 "complex molecule" validation).
LARGE_COMPLEX_REFS = {
    "curcumin", "celecoxib", "sildenafil", "losartan",
    "atorvastatin", "simvastatin", "tamoxifen",
    "fluoxetine", "imipramine", "testosterone",
}


def canonical(smiles: str) -> str:
    """Return canonical SMILES or empty string on failure."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else ""


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_file(url: str, dest: str) -> str:
    """Download url to dest, skip if exists. Returns dest path."""
    if os.path.exists(dest):
        print(f"  [skip] Already exists: {dest}")
        return dest
    print(f"  Downloading {url} ...")
    # Try curl first (handles macOS SSL better), fall back to urllib with unverified context
    try:
        subprocess.run(["curl", "-sL", "-o", dest, url], check=True, timeout=60)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ctx) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    print(f"  Saved to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------
def load_esol(raw_path: str) -> pd.DataFrame:
    """Load ESOL (Delaney) dataset. Returns DataFrame with [smiles, logS, compound_id]."""
    df = pd.read_csv(raw_path)
    out = pd.DataFrame()
    out["smiles"] = df["smiles"]
    out["logS"] = df["measured log solubility in mols per litre"]
    out["compound_id"] = df["Compound ID"]
    # Drop rows with missing smiles
    out = out.dropna(subset=["smiles"]).reset_index(drop=True)
    return out


def load_freesolv(raw_path: str) -> pd.DataFrame:
    """Load FreeSolv dataset (semicolon-delimited with comments)."""
    rows = []
    with open(raw_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 5:
                continue
            try:
                rows.append({
                    "compound_id": parts[0],
                    "smiles": parts[1],
                    "iupac_name": parts[2],
                    "hydration_free_energy": float(parts[3]),
                    "expt_uncertainty": float(parts[4]),
                })
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(rows)
    return df


def load_lipophilicity(raw_path: str) -> pd.DataFrame:
    """Load Lipophilicity dataset. Returns DataFrame with [smiles, logD, chembl_id]."""
    df = pd.read_csv(raw_path)
    out = pd.DataFrame()
    out["smiles"] = df["smiles"]
    out["logD"] = df["exp"]
    out["chembl_id"] = df["CMPD_CHEMBLID"]
    out = out.dropna(subset=["smiles"]).reset_index(drop=True)
    return out


def load_qm8(raw_path: str) -> pd.DataFrame:
    """Load QM8 dataset. Returns DataFrame with SMILES + 12 electronic spectra properties.

    Properties: excitation energies (E1, E2) and oscillator strengths (f1, f2)
    computed with 3 methods: CC2, PBE0, CAM-B3LYP.

    Note: The original CSV has columns PBE0.1 which are known duplicates of PBE0 (bug).
    We drop the duplicates.
    """
    df = pd.read_csv(raw_path)
    # Drop duplicate columns (known bug: PBE0.1 == PBE0)
    dup_cols = [c for c in df.columns if ".1" in c]
    df = df.drop(columns=dup_cols)
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Molecule Scanning & Selection
# ---------------------------------------------------------------------------
def build_canonical_index(df: pd.DataFrame) -> dict:
    """Build a dict mapping canonical SMILES -> row index for a DataFrame."""
    index = {}
    for i, smi in enumerate(df["smiles"]):
        can = canonical(str(smi))
        if can:
            index[can] = i
    return index


def scan_known_molecules(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    Scan a dataset for well-known molecules.
    Returns dict: molecule_name -> (row_index, canonical_smiles, property_values).
    """
    can_index = build_canonical_index(df)
    found = {}
    non_prop = {"smiles", "compound_id", "chembl_id", "iupac_name", "expt_uncertainty"}
    prop_cols = [c for c in df.columns if c not in non_prop]

    print(f"\n  Scanning {dataset_name} ({len(df):,} molecules) for known molecules...")
    for name, smi in KNOWN_MOLECULES.items():
        can = canonical(smi)
        if can in can_index:
            idx = can_index[can]
            props = {col: df[col].iloc[idx] for col in prop_cols}
            found[name] = {"index": idx, "canonical_smiles": can, "properties": props}
            prop_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in props.items())
            print(f"    FOUND: {name:20s} (row {idx:>6d}) | {prop_str}")

    if not found:
        print(f"    No known molecules found in {dataset_name}.")
    return found


# ---------------------------------------------------------------------------
# CSV Generation
# ---------------------------------------------------------------------------
def create_reference_csv(df: pd.DataFrame, ref_idx: int, output_path: str, ref_name: str):
    """Move reference molecule to row 0 and save CSV."""
    ref_row = df.iloc[[ref_idx]]
    other_rows = df.drop(index=ref_idx)
    out_df = pd.concat([ref_row, other_rows], ignore_index=True)
    out_df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path} ({len(out_df):,} molecules, ref={ref_name} in row 0)")


# ---------------------------------------------------------------------------
# QM9 Additional References
# ---------------------------------------------------------------------------
def prepare_qm9_additional_references():
    """Create additional QM9 CSVs with different reference molecules."""
    qm9_path = os.path.join("data", "qm9_anilin.csv")
    if not os.path.exists(qm9_path):
        print("  [skip] QM9 file not found, skipping additional references")
        return

    df = pd.read_csv(qm9_path)
    can_index = build_canonical_index(df)

    qm9_refs = {
        "pyridine": "c1ccncc1",
        "toluene": "Cc1ccccc1",
    }

    for name, smi in qm9_refs.items():
        can = canonical(smi)
        if can in can_index:
            idx = can_index[can]
            out_path = os.path.join(BENCHMARK_DIR, f"qm9_{name}.csv")
            create_reference_csv(df, idx, out_path, name)
        else:
            print(f"    WARNING: {name} ({smi}) not found in QM9 dataset")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    scan_only = "--scan-only" in sys.argv

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    print("=" * 70)
    print("  MSI Benchmark Data Preparation")
    print("=" * 70)

    # ---- Step 1: Download raw files ----
    print("\n--- Step 1: Download raw datasets ---")
    raw_paths = {}
    for name, url in URLS.items():
        ext = ".txt" if name == "freesolv" else ".csv"
        dest = os.path.join(RAW_DIR, f"{name}_raw{ext}")
        raw_paths[name] = download_file(url, dest)

    # ---- Step 2: Load and normalize ----
    print("\n--- Step 2: Load and normalize datasets ---")
    datasets = {
        "esol": load_esol(raw_paths["esol"]),
        "freesolv": load_freesolv(raw_paths["freesolv"]),
        "lipophilicity": load_lipophilicity(raw_paths["lipophilicity"]),
        "qm8": load_qm8(raw_paths["qm8"]),
    }
    for name, df in datasets.items():
        print(f"  {name}: {len(df):,} molecules, columns: {list(df.columns)}")

    # ---- Step 3: Scan for known molecules ----
    print("\n--- Step 3: Scan for known reference molecules ---")
    scan_results = {}
    for name, df in datasets.items():
        scan_results[name] = scan_known_molecules(df, name)

    if scan_only:
        print("\n[scan-only mode] Done. Review the results above to select reference molecules.")
        return

    # ---- Step 4: Generate benchmark CSVs ----
    print("\n--- Step 4: Generate benchmark CSVs ---")

    # ESOL references
    esol_refs = {
        "caffeine": "caffeine",
        "naphthalene": "naphthalene",
    }
    # Add phenol if found
    if "phenol" in scan_results["esol"]:
        esol_refs["phenol"] = "phenol"

    for ref_label, mol_name in esol_refs.items():
        if mol_name in scan_results["esol"]:
            idx = scan_results["esol"][mol_name]["index"]
            out_path = os.path.join(BENCHMARK_DIR, f"esol_{ref_label}.csv")
            create_reference_csv(datasets["esol"], idx, out_path, ref_label)
        else:
            print(f"    WARNING: {mol_name} not found in ESOL, skipping")

    # FreeSolv references
    freesolv_refs = {
        "toluene": "toluene",
        "ethanol": "ethanol",
    }
    # Add acetone if found
    if "acetone" in scan_results["freesolv"]:
        freesolv_refs["acetone"] = "acetone"
    elif "aniline" in scan_results["freesolv"]:
        freesolv_refs["aniline"] = "aniline"

    for ref_label, mol_name in freesolv_refs.items():
        if mol_name in scan_results["freesolv"]:
            idx = scan_results["freesolv"][mol_name]["index"]
            out_path = os.path.join(BENCHMARK_DIR, f"freesolv_{ref_label}.csv")
            create_reference_csv(datasets["freesolv"], idx, out_path, ref_label)
        else:
            print(f"    WARNING: {mol_name} not found in FreeSolv, skipping")

    # Lipophilicity references — include drugs of varied size/complexity
    # to answer Reviewer 1 point 4 (broader validation across complex molecules).
    lipo_found = scan_results["lipophilicity"]

    # Preferred mix: keep the two existing drug refs and add any of the
    # large/complex references found in the dataset.
    lipo_priority = [
        # medium (already in prior benchmark runs)
        "diclofenac", "naproxen",
        # large / complex — answering point 4
        "celecoxib", "sildenafil", "losartan", "atorvastatin",
        "simvastatin", "tamoxifen", "fluoxetine", "imipramine",
        # fallbacks
        "ibuprofen", "acetaminophen", "metoprolol",
    ]
    lipo_refs_selected = [m for m in lipo_priority if m in lipo_found]

    # If still < 2 refs, fall back to a median-logD molecule.
    if len(lipo_refs_selected) < 2:
        lipo_df = datasets["lipophilicity"]
        median_logD = lipo_df["logD"].median()
        median_idx = (lipo_df["logD"] - median_logD).abs().idxmin()
        out_path = os.path.join(BENCHMARK_DIR, "lipo_median.csv")
        create_reference_csv(lipo_df, median_idx, out_path, "median_logD")

    for mol_name in lipo_refs_selected:
        idx = lipo_found[mol_name]["index"]
        out_path = os.path.join(BENCHMARK_DIR, f"lipo_{mol_name}.csv")
        create_reference_csv(datasets["lipophilicity"], idx, out_path, mol_name)

    # QM8 references (electronic spectra — quantum dataset)
    qm8_refs = {
        "aniline": "aniline",
        "pyridine": "pyridine",
        "phenol": "phenol",
    }
    print("\n  QM8 references:")
    for ref_label, mol_name in qm8_refs.items():
        if mol_name in scan_results["qm8"]:
            idx = scan_results["qm8"][mol_name]["index"]
            out_path = os.path.join(BENCHMARK_DIR, f"qm8_{ref_label}.csv")
            create_reference_csv(datasets["qm8"], idx, out_path, ref_label)
        else:
            print(f"    WARNING: {mol_name} not found in QM8, skipping")

    # QM9 additional references
    print("\n  QM9 additional references:")
    prepare_qm9_additional_references()

    # ---- Step 5: Summary ----
    print("\n--- Step 5: Summary ---")
    benchmark_files = [f for f in os.listdir(BENCHMARK_DIR) if f.endswith(".csv")]
    print(f"  Generated {len(benchmark_files)} benchmark CSVs in {BENCHMARK_DIR}/:")
    for f in sorted(benchmark_files):
        path = os.path.join(BENCHMARK_DIR, f)
        n_rows = sum(1 for _ in open(path)) - 1  # subtract header
        print(f"    {f}: {n_rows:,} molecules")

    print(f"\n  Next step: Run benchmark.ipynb")
    print("=" * 70)


if __name__ == "__main__":
    main()
