"""
test_grid_infrastructure.py — smoke tests for the grid benchmark pipeline.

Ensures the core modules (representations, similarity_measures,
benchmark_matrix) import cleanly and that the 12-representation catalog is
internally consistent.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def test_representations_imports():
    import representations
    assert hasattr(representations, "ALL_REPRESENTATIONS")
    assert hasattr(representations, "generate_representation")


def test_twelve_representations_catalogued():
    from representations import ALL_REPRESENTATIONS
    expected = {
        "ecfp4_1024", "ecfp4_2048", "ecfp4_4096", "ecfp6_2048",
        "maccs", "rdkit_fp",
        "rdkit_physchem",
        "mol2vec", "unimol", "hybrid",
        "chemberta", "chemberta_77m",
    }
    assert set(ALL_REPRESENTATIONS.keys()) == expected, \
        f"Representations drifted: {sorted(ALL_REPRESENTATIONS)}"


def test_similarity_measures_imports():
    import similarity_measures as sm
    assert hasattr(sm, "COMPATIBLE_METRICS")
    assert hasattr(sm, "METRIC_ORIENTATION")


def test_seven_metrics_catalogued():
    from similarity_measures import METRIC_ORIENTATION
    expected = {"tanimoto", "tversky_2_8", "tversky_8_2", "dice",
                "cosine", "mahalanobis_d", "mahalanobis_theta"}
    assert set(METRIC_ORIENTATION.keys()) == expected


def test_compatibility_rules():
    from similarity_measures import COMPATIBLE_METRICS
    # fingerprints take every metric; descriptors/embeddings only continuous
    fp_metrics = set(COMPATIBLE_METRICS["fingerprint"])
    assert "tanimoto" in fp_metrics
    assert "mahalanobis_theta" in fp_metrics

    for fam in ("descriptor", "embedding"):
        fam_metrics = set(COMPATIBLE_METRICS[fam])
        assert "tanimoto" not in fam_metrics
        assert "mahalanobis_theta" in fam_metrics


def test_benchmark_matrix_imports():
    import benchmark_matrix
    assert hasattr(benchmark_matrix, "DatasetCase")
    assert hasattr(benchmark_matrix, "run_grid")
    assert hasattr(benchmark_matrix, "win_rate_table")


def test_tversky_generalises_tanimoto():
    """Tversky(alpha=1, beta=1) must equal Tanimoto on the same bits."""
    import numpy as np
    from similarity_measures import tanimoto_bits, tversky_bits
    rng = np.random.default_rng(42)
    for _ in range(20):
        a = rng.integers(0, 2, size=128, dtype=np.uint8)
        b = rng.integers(0, 2, size=128, dtype=np.uint8)
        t = tanimoto_bits(a, b)
        tv = tversky_bits(a, b, alpha=1.0, beta=1.0)
        assert abs(t - tv) < 1e-9, f"Tversky(1,1) != Tanimoto: {t} vs {tv}"


def test_mahalanobis_angle_self_is_zero():
    """Mahalanobis angle of a vector with itself must be 0."""
    import numpy as np
    from similarity_measures import mahalanobis_angle_matrix
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(50, 10))
    angles = mahalanobis_angle_matrix(mat, ref_idx=0)
    assert abs(angles[0]) < 1e-6, f"Self-angle should be 0, got {angles[0]}"


def test_grid_results_file_structure():
    """If grid_results.csv exists, it has the expected columns and 12 reps."""
    import pandas as pd
    path = os.path.join(os.path.dirname(HERE), "output_grid", "grid_results.csv")
    if not os.path.exists(path):
        return  # skip on clean clone
    df = pd.read_csv(path)
    expected_cols = {"dataset", "reference", "case", "property", "rep",
                     "rep_family", "metric", "n", "deviation",
                     "deviation_norm", "p_value", "n_molecules"}
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns in grid_results: {missing}"
    n_reps = df["rep"].nunique()
    assert n_reps == 12, f"Expected 12 reps, got {n_reps}: {sorted(df.rep.unique())}"


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed.append(fn.__name__)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
