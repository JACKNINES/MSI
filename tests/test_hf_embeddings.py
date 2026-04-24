"""
test_hf_embeddings.py — smoke tests for the HuggingFace embedding integration.

Cover imports, configuration integrity, and the end-to-end shape of a small
inference without downloading gigabytes of model weights. Run with:

    cd /Users/nose/Desktop/MSI
    python3 -m pytest tests/ -v
    # or without pytest:
    python3 tests/test_hf_embeddings.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MSI_ROOT = os.path.dirname(HERE)
sys.path.insert(0, MSI_ROOT)


# ----------------------------------------------------------------------------
# Smoke tests that always pass (no network, no GPU, no model download)
# ----------------------------------------------------------------------------

def test_hf_embeddings_imports():
    """hf_embeddings module imports cleanly without side effects."""
    import hf_embeddings
    assert hasattr(hf_embeddings, "MODELS")
    assert hasattr(hf_embeddings, "generate_hf_vectors")


def test_hf_models_catalog():
    """The MODELS catalog has both chemberta variants with expected dims."""
    from hf_embeddings import MODELS
    assert "chemberta" in MODELS
    assert "chemberta_77m" in MODELS
    assert MODELS["chemberta"]["dim"] == 768
    assert MODELS["chemberta_77m"]["dim"] == 384
    # MolFormer is explicitly dropped (incompatible with transformers >=4.40)
    assert "molformer" not in MODELS


def test_representations_catalog():
    """representations.py exposes the new hf_cached embeddings."""
    from representations import ALL_REPRESENTATIONS, EMBEDDING_TYPES
    assert "chemberta" in EMBEDDING_TYPES
    assert "chemberta_77m" in EMBEDDING_TYPES
    # Both live in the embedding family
    for name in ("chemberta", "chemberta_77m"):
        assert name in ALL_REPRESENTATIONS
        assert EMBEDDING_TYPES[name]["kind"] == "hf_cached"


def test_hf_vectors_cache_reuse(tmp_path=None):
    """generate_hf_vectors skips inference if a fresh CSV already exists."""
    from hf_embeddings import generate_hf_vectors

    # Build a tiny fake input CSV and a fake cached vectors CSV
    tmp = tempfile.mkdtemp()
    input_csv = os.path.join(tmp, "tiny.csv")
    pd.DataFrame({"smiles": ["CCO", "CCC", "c1ccccc1"]}).to_csv(input_csv, index=False)

    # Pre-seed the output with valid-looking vectors (no NaN)
    out_name = "tiny_chemberta_768d_vectors.csv"
    out_path = os.path.join(tmp, out_name)
    cached = np.random.RandomState(0).rand(3, 768).astype(np.float32)
    pd.DataFrame(cached, columns=[str(i) for i in range(768)]).to_csv(out_path, index=False)

    # Call generate_hf_vectors — it should return the cached file
    # without importing torch or downloading anything.
    returned = generate_hf_vectors(
        path=input_csv,
        model_key="chemberta",
        output_dir=tmp,
        output_name=out_name,
        use_cuda=False,
    )
    assert returned == out_path
    # File was not overwritten
    read_back = pd.read_csv(returned).values
    np.testing.assert_array_almost_equal(read_back, cached, decimal=5)


def test_dataset_paths_all_present():
    """generate_hf_all.dataset_paths() returns the 15 expected benchmark cases."""
    # This test assumes the benchmark CSVs have been prepared
    bench = os.path.join(MSI_ROOT, "data", "benchmark")
    if not os.path.isdir(bench):
        # Skip if benchmark data not yet prepared (clean clone)
        return
    from generate_hf_all import dataset_paths
    paths = dataset_paths()
    stems = {os.path.splitext(os.path.basename(p))[0] for p in paths}
    expected = {
        "qm9_anilin", "qm9_pyridine",
        "qm8_aniline", "qm8_pyridine", "qm8_phenol",
        "esol_caffeine", "esol_naphthalene",
        "freesolv_toluene", "freesolv_ethanol",
        "lipo_diclofenac", "lipo_naproxen", "lipo_celecoxib",
        "lipo_simvastatin", "lipo_fluoxetine", "lipo_imipramine",
    }
    missing = expected - stems
    assert not missing, f"Faltan casos del benchmark: {missing}"


def test_onnx_stub_monkey_patch_signature():
    """The monkey-patch for transformers.onnx must be present in hf_embeddings."""
    import hf_embeddings
    import inspect
    source = inspect.getsource(hf_embeddings.generate_hf_vectors)
    assert "transformers.onnx" in source, \
        "Missing transformers.onnx monkey-patch (needed for MolFormer-like models)"


# ----------------------------------------------------------------------------
# Optional heavy smoke test (real model download) — skipped by default
# ----------------------------------------------------------------------------

def test_chemberta_real_small_inference():
    """End-to-end test on 3 SMILES — only runs if RUN_HF_HEAVY_TESTS=1 env set.

    Downloads seyonec/ChemBERTa-zinc-base-v1 (~50 MB) and runs inference on CPU.
    Takes ~30 seconds first run (download), <1s cached.
    """
    if os.environ.get("RUN_HF_HEAVY_TESTS") != "1":
        return  # skip by default

    from hf_embeddings import generate_hf_vectors
    tmp = tempfile.mkdtemp()
    input_csv = os.path.join(tmp, "smoke.csv")
    pd.DataFrame({"smiles": [
        "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
        "c1ccc(cc1)N",              # aniline
    ]}).to_csv(input_csv, index=False)

    out = generate_hf_vectors(
        path=input_csv, model_key="chemberta",
        output_dir=tmp, use_cuda=False, batch_size=3,
    )
    mat = pd.read_csv(out).values
    assert mat.shape == (3, 768)
    assert not np.isnan(mat).any(), "Embedding contains NaN"


# ----------------------------------------------------------------------------
# Runner — works without pytest
# ----------------------------------------------------------------------------

def main():
    tests = [fn for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed.append(fn.__name__)
    print()
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
