"""
MSI Workflow Runner
===================

Executes the complete Mahalanobis Similarity Index (MSI) pipeline on all
available datasets from the MMSI project.

Datasets processed
------------------
**Paper case studies** (from prueba.py):
  - Aspirin (ACI dataset, ~34k molecules)
  - Ibuprofen (ACI dataset, ~34k molecules)
  - Curcumin (ACI dataset, ~34k molecules)
  - Aniline (QM9 dataset, ~134k molecules with quantum-chemical properties)

**NSAID extension** (from prueba2.py):
  - Fenoprofen (~1.8k PubChem molecules)
  - Flurbiprofen (~4.6k PubChem molecules)
  - Ketoprofen (~1.4k PubChem molecules)
  - Naproxen (~2.9k PubChem molecules)

Usage
-----
    python workflow.py

Requirements
------------
  - ``model_300dim.pkl`` in the same directory (or parent MMSI directory)
  - Input CSVs in ``data/`` directory (configurable via DATA_DIR)
  - Creates ``output/`` subdirectory for all results

Authors: Emiliano Montoya & Elliot Ridout
Restructured: 2026
"""

import os
import time

from gensim.models import Word2Vec

from msi import (
    generate_2d_vectors,
    generate_300d_vectors,
    analyze_against_reference,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Model path: in data/ directory
MODEL_PATH = os.path.join(DATA_DIR, "model_300dim.pkl")

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
# Each entry describes one dataset to process:
#   name             : Human-readable identifier
#   filename         : CSV filename in DATA_DIR
#   preserve_columns : Whether to keep all original CSV columns in 2D output
#   description      : Short description for console output
# ---------------------------------------------------------------------------

DATASETS = [
    # ---- Paper Case Study I: ACI datasets (simple SMILES-only CSVs) ---------
    {
        "name": "aspirin",
        "filename": "aspirin.csv",
        "preserve_columns": False,
        "description": "ACI aspirin dataset (~34k molecules)",
    },
    {
        "name": "ibuprofen",
        "filename": "ibuprofen.csv",
        "preserve_columns": False,
        "description": "ACI ibuprofen dataset (~34k molecules)",
    },
    {
        "name": "curcumin",
        "filename": "curcumin.csv",
        "preserve_columns": False,
        "description": "ACI curcumin dataset (~34k molecules)",
    },
    # ---- Paper Case Study II: QM9 dataset -----------------------------------
    {
        "name": "qm9_anilin",
        "filename": "qm9_anilin.csv",
        "preserve_columns": True,   # Keep HOMO, LUMO, gap, etc.
        "description": "QM9 aniline subset (~134k molecules with quantum properties)",
    },
    # ---- NSAID extension (raw PubChem exports) ------------------------------
    {
        "name": "fenoprofen",
        "filename": "fenoprofen.csv",
        "preserve_columns": True,   # Keep cmpdname, iupacname, etc.
        "description": "NSAID fenoprofen PubChem dataset",
    },
    {
        "name": "flurbiprofen",
        "filename": "flurbiprofen.csv",
        "preserve_columns": True,
        "description": "NSAID flurbiprofen PubChem dataset",
    },
    {
        "name": "ketoprofen",
        "filename": "ketoprofen.csv",
        "preserve_columns": True,
        "description": "NSAID ketoprofen PubChem dataset",
    },
    {
        "name": "naproxen",
        "filename": "naproxen.csv",
        "preserve_columns": True,
        "description": "NSAID naproxen PubChem dataset",
    },
]


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def run_dataset_pipeline(
    dataset_config: dict,
    model: Word2Vec,
    data_dir: str,
    output_dir: str,
) -> dict:
    """
    Run the full vectorization + similarity pipeline for a single dataset.

    Steps:
      1. ``generate_2d_vectors``  → ``<name>_2d_vectors.csv``
      2. ``generate_300d_vectors`` → ``<name>_300d_vectors.csv``
      3. ``analyze_against_reference`` → ``<name>_2d_vectors_against_it.csv``

    Parameters
    ----------
    dataset_config : dict
        Dataset configuration with keys:
        ``name``, ``filename``, ``preserve_columns``, ``description``.
    model : Word2Vec
        Pre-trained mol2vec model.
    data_dir : str
        Directory containing the input CSV.
    output_dir : str
        Directory for output files.

    Returns
    -------
    dict
        Summary with keys:
        ``name``, ``description``, ``path_2d``, ``path_300d``,
        ``path_results``, ``n_molecules``, ``elapsed_seconds``.
    """
    input_path = os.path.join(data_dir, dataset_config["filename"])
    name = dataset_config["name"]
    t0 = time.time()

    # Step 1: Generate 2D vectors
    print(f"\n  [1/3] Generating 2D vectors for {name}...")
    path_2d = generate_2d_vectors(
        path=input_path,
        model=model,
        output_dir=output_dir,
        preserve_columns=dataset_config["preserve_columns"],
    )

    # Step 2: Generate 300D vectors
    print(f"  [2/3] Generating 300D vectors for {name}...")
    path_300d = generate_300d_vectors(
        path=input_path,
        model=model,
        output_dir=output_dir,
    )

    # Step 3: Run similarity analysis
    print(f"  [3/3] Running similarity analysis for {name}...")
    results_df, metrics = analyze_against_reference(
        path_2d=path_2d,
        path_300d=path_300d,
        output_dir=output_dir,
    )

    elapsed = time.time() - t0

    return {
        "name": name,
        "description": dataset_config["description"],
        "path_2d": path_2d,
        "path_300d": path_300d,
        "path_results": os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(path_2d))[0]}_against_it.csv",
        ),
        "n_molecules": len(results_df),
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }


def print_summary(results: list) -> None:
    """
    Print a formatted summary table of all pipeline results.

    Parameters
    ----------
    results : list of dict
        Each dict returned by :func:`run_dataset_pipeline`.
    """
    print("\n")
    print("=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(
        f"{'Dataset':<20} {'Molecules':>10} {'Time (s)':>10} "
        f"{'Cov. Cond #':>14} {'Tanimoto Range':>18}"
    )
    print("-" * 80)

    total_time = 0.0
    total_mols = 0

    for r in results:
        m = r.get("metrics", {})
        cond = m.get("covariance_condition_number", "N/A")
        tani = m.get("tanimoto_range", (None, None))

        cond_str = f"{cond:.2e}" if isinstance(cond, (int, float)) else str(cond)
        tani_str = (
            f"[{tani[0]:.4f}, {tani[1]:.4f}]"
            if tani[0] is not None
            else "N/A"
        )

        print(
            f"{r['name']:<20} {r['n_molecules']:>10,} {r['elapsed_seconds']:>10.1f} "
            f"{cond_str:>14} {tani_str:>18}"
        )

        total_time += r["elapsed_seconds"]
        total_mols += r["n_molecules"]

    print("-" * 80)
    print(f"{'TOTAL':<20} {total_mols:>10,} {total_time:>10.1f}")
    print("=" * 80)

    print(f"\nOutput files saved to: {os.path.abspath(results[0]['path_2d']).rsplit('/', 1)[0]}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Entry point: load the mol2vec model and run the pipeline on all datasets.
    """
    print("=" * 80)
    print("  MSI WORKFLOW RUNNER")
    print("  Mahalanobis Similarity Index — Full Pipeline")
    print("=" * 80)

    # --- Step 1: Create output directory -------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory : {os.path.abspath(OUTPUT_DIR)}")
    print(f"Data directory   : {os.path.abspath(DATA_DIR)}")
    print(f"Model path       : {os.path.abspath(MODEL_PATH)}")

    # --- Step 2: Load pre-trained mol2vec model ------------------------------
    if not os.path.exists(MODEL_PATH):
        print(
            f"\nERROR: Model file not found at '{MODEL_PATH}'.\n"
            f"Please place 'model_300dim.pkl' in:\n"
            f"  - {SCRIPT_DIR}  (local), or\n"
            f"  - {DATA_DIR}  (data folder)\n"
        )
        return

    print(f"\nLoading model...")
    t_model = time.time()
    model = Word2Vec.load(MODEL_PATH)
    print(f"Model loaded in {time.time() - t_model:.1f}s. Vocabulary size: {len(model.wv):,}")

    # --- Step 3: Process each dataset ----------------------------------------
    results = []
    available = [
        ds for ds in DATASETS
        if os.path.exists(os.path.join(DATA_DIR, ds["filename"]))
    ]
    skipped = [
        ds for ds in DATASETS
        if not os.path.exists(os.path.join(DATA_DIR, ds["filename"]))
    ]

    if skipped:
        print(f"\nWARNING: {len(skipped)} dataset(s) not found and will be skipped:")
        for ds in skipped:
            print(f"  - {ds['filename']} ({ds['description']})")

    print(f"\nProcessing {len(available)} dataset(s)...\n")

    for i, dataset in enumerate(available, 1):
        print(f"{'=' * 80}")
        print(f"[{i}/{len(available)}] {dataset['description']}")
        print(f"  File: {dataset['filename']}")
        print(f"{'=' * 80}")

        try:
            result = run_dataset_pipeline(dataset, model, DATA_DIR, OUTPUT_DIR)
            results.append(result)
            print(f"\n  Done in {result['elapsed_seconds']:.1f}s "
                  f"({result['n_molecules']:,} molecules)")
        except Exception as exc:
            print(f"\n  ERROR processing {dataset['name']}: {exc}")
            continue

    # --- Step 4: Print summary -----------------------------------------------
    if results:
        print_summary(results)
    else:
        print("\nNo datasets were processed successfully.")


if __name__ == "__main__":
    main()
