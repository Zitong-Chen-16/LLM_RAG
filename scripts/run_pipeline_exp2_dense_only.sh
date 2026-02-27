#!/usr/bin/env bash

uv run src/process/query_ppl.py \
  --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 \
  --out data/answers/leaderboard_answers_v2_dense_only.json \
  --bm25_dir indexes/bm25_v3 \
  --dense_dir indexes/dense_gte-Qwen2-7B-instruct_v2 \
  --embed_model Alibaba-NLP/gte-Qwen2-7B-instruct \
  --embed_quant_backend 8bit \
  --quant_backend gptq \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 100 \
  --k_ctx 20 \
  --max_context_tokens 24000 \
  --k_retrieve 100 \
  --k_dense 200 \
  --k_sparse 200 \
  --fusion_method rrf \
  --rrf_k 60 \
  --device cuda:0 \
  --reader_device cuda:1 \
  --dedup_doc \
  --stage1_k 60 \
  --mmr_lambda 0.75 \
  --w_dense 1.0 \
  --w_sparse 0.0 \
  --retrieval_mode dense
