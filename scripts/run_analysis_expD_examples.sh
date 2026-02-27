#!/usr/bin/env bash

uv run src/analysis/prepare_exp_d_examples.py \
  --queries leaderboard_queries.json \
  --details_jsonl analysis/exp_ab/retrieval_details.jsonl \
  --closed_answers data/answers/leaderboard_answers_v1_closed_book.json \
  --dense_answers data/answers/leaderboard_answers_v2_dense_only.json \
  --sparse_answers data/answers/leaderboard_answers_v3_sparse_only.json \
  --hybrid_answers data/answers/leaderboard_answers_v4.json \
  --hybrid_minmax_answers data/answers/leaderboard_answers_v5_hybrid_minmax.json \
  --faithfulness_csv analysis/exp_c/faithfulness_sheet.csv \
  --out_json analysis/exp_d/selected_examples.json \
  --out_md analysis/exp_d/representative_examples.md \
  --n_per_bucket 2
