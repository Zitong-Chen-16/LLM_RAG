#!/usr/bin/env bash
set -euo pipefail

uv run src/analysis/summarize_analysis_results.py \
  --exp_ab_dir analysis/exp_ab \
  --exp_ab_exp8_dir analysis/exp_ab_exp8_minilm \
  --exp_d_dir analysis/exp_d \
  --queries leaderboard_queries.json \
  --out_dir analysis/summary_report

echo "[run_analysis_summary] outputs in analysis/summary_report"
