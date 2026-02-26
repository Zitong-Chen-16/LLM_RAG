#!/usr/bin/env bash
uv run src/process/query_pipeline.py \
  --model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 48 \
  --k_ctx 4 \
  --max_context_tokens 3000 \
  --k_retrieve 60 \
  --k_dense 200 \
  --k_sparse 200 \
  --fusion_method rrf \
  --rrf_k 60 \
  --device cuda:1
#   --dedup_doc \
#   --stage1_k 24 \
#   --mmr_lambda 0.75 \
  
