#!/usr/bin/env bash

uv run src/analysis/run_exp_ab_retrieval_analysis.py \
  --queries leaderboard_queries.json \
  --chunks data/processed/chunks.jsonl \
  --out_dir analysis/exp_ab_exp8_minilm \
  --methods hybrid_rrf \
  --k 10 \
  --bm25_dir indexes/bm25_v3 \
  --dense_dir indexes/dense_all-MiniLM-L6-v2_v2 \
  --embed_model sentence-transformers/all-MiniLM-L6-v2 \
  --embed_quant_backend none \
  --device cuda:0 \
  --w_dense 0.5 \
  --w_sparse 0.5 \
  --k_dense 200 \
  --k_sparse 200 \
  --rrf_k 60 \
  --sparse_title_weight 3 \
  --sparse_heading_weight 2 \
  --sparse_body_weight 1 \
  --sparse_add_bigrams \
  --sparse_prf \
  --sparse_prf_k 8 \
  --sparse_prf_terms 6 \
  --sparse_prf_alpha 0.65
