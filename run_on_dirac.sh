#!/usr/bin/env bash
# run_on_dirac.sh — full representation × metric grid on the dirac server.
#
# Assumes:
#   * you are in the MSI repo root
#   * CUDA is available (checked below)
#   * data/ holds model_300dim.pkl, qm9_anilin.csv, benchmark/*.csv, etc.
#   * optional: output_benchmark/ with cached mol2vec 300D CSVs (saves ~30 min)
#
# Runtime: ~30-60 min end-to-end on a GPU machine (QM9 Uni-Mol is the long pole).

set -euo pipefail

echo "=== Environment sanity check ==="
python3 -c "
import torch, rdkit, gensim, mol2vec
print(f'  torch {torch.__version__}  CUDA={torch.cuda.is_available()}  devices={torch.cuda.device_count()}')
print(f'  rdkit {rdkit.__version__}, gensim {gensim.__version__}')
import unimol_tools
print('  unimol-tools ok')
"

echo
echo "=== Step 1: prepare benchmark CSVs =========================="
python3 prepare_benchmark.py

echo
echo "=== Step 2: Tier-1 grid (fingerprints + descriptors + mol2vec) "
echo "           on ALL cases including QM9 ========================"
python3 -u run_grid.py --qm9 --boot 1000

echo
echo "=== Step 3: Generate Uni-Mol vectors for EVERY case ========="
# Runs the old Uni-Mol notebook's extraction step on every benchmark CSV
# (including QM9, which we skipped locally). Uses CUDA automatically.
python3 - <<'PY'
import os, glob, time
from unimol_msi import generate_unimol_vectors

BENCHMARK_DIR = 'data/benchmark'
UNIMOL_OUTPUT = 'output_unimol'
DATA_DIR      = 'data'

datasets = sorted(glob.glob(f'{BENCHMARK_DIR}/*.csv'))
# Also include data/qm9_anilin.csv (original aniline reference)
datasets.append(os.path.join(DATA_DIR, 'qm9_anilin.csv'))

for path in datasets:
    stem = os.path.splitext(os.path.basename(path))[0]
    out_name = f'{stem}_unimol_512d_vectors.csv'
    out_path = os.path.join(UNIMOL_OUTPUT, out_name)
    if os.path.exists(out_path):
        print(f'  [skip] {stem} (cached)')
        continue
    print(f'\n--- {stem} ---')
    t0 = time.time()
    generate_unimol_vectors(
        path=path,
        output_dir=UNIMOL_OUTPUT,
        output_name=out_name,
        batch_size=256,          # big batch on GPU
        use_cuda=True,
    )
    print(f'  done in {(time.time()-t0)/60:.1f} min')
PY

echo
echo "=== Step 4: Tier-2 grid (+ Uni-Mol + hybrid) on ALL cases ==="
python3 -u run_grid.py --qm9 --tier2 --boot 1000

echo
echo "=== Done. Artefacts in output_grid/: ========================"
ls -la output_grid/
echo
echo "Key deliverables:"
echo "  output_grid/grid_results.csv               — raw per-row results"
echo "  output_grid/win_rate_vs_ecfp4_tanimoto.csv — ranked (rep, metric) pairs"
echo "  output_grid/summary_by_family.csv          — mean perf by family"
echo "  output_grid/final_representation_x_metric.csv — paper-ready table"
echo "  output_grid/heatmap_norm_deviation.png     — figure for the paper"
echo "  output_grid/heatmap_by_domain.png          — electronic vs experimental"
