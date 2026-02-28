#!/usr/bin/env bash
set -euo pipefail

# 1) Main A/B diagnostics for sparse/dense/hybrid variants
./scripts/run_analysis_expA_B.sh

# 2) A/B diagnostics for exp8 hybrid (MiniLM embedder), used by Exp D
./scripts/run_analysis_expA_B_exp8_minilm.sh

# 3) Example packet with exp8 embedding-ablation bucket
./scripts/run_analysis_expD_examples.sh

echo "[analysis_minimal] done. Outputs:"
echo "  - analysis/exp_ab/"
echo "  - analysis/exp_ab_exp8_minilm/"
echo "  - analysis/exp_d/"
