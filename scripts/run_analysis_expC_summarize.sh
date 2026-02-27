#!/usr/bin/env bash

uv run src/analysis/summarize_exp_c_faithfulness.py \
  --in_csv analysis/exp_c/faithfulness_sheet.csv \
  --out_json analysis/exp_c/faithfulness_summary.json
