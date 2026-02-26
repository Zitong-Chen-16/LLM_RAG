#!/usr/bin/env bash
uv run src/process/query_ppl.py \
  --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 \
  --quant_backend gptq \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 48 \
  --k_ctx 6 \
  --max_context_tokens 24000 \
  --k_retrieve 60 \
  --k_dense 100 \
  --k_sparse 100 \
  --fusion_method rrf \
  --rrf_k 60 \
  --device cuda:0 \
  --reader_device cuda:1 \
  --dedup_doc \
  --stage1_k 60 \
  --mmr_lambda 0.75 \
  
