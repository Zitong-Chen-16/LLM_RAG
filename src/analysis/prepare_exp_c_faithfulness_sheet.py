#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]
PROCESS_DIR = ROOT / "src" / "process"
if str(PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESS_DIR))
if str(THIS.parent) not in sys.path:
    sys.path.insert(0, str(THIS.parent))

from common import (  # noqa: E402
    classify_question_type,
    load_answers,
    load_queries,
    snippet,
    stratified_sample,
    write_csv,
)
from hybrid_retrieval import build_retriever  # noqa: E402
from query_ppl import mmr_select_chunk_ids  # noqa: E402
from utils import load_chunk_text_map  # noqa: E402


def _answer_for(qid: str, answers: Dict[str, str]) -> str:
    return str(answers.get(str(qid), "") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Experiment C manual faithfulness sheet")
    ap.add_argument("--queries", default="leaderboard_queries.json")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--closed_answers", required=True)
    ap.add_argument("--rag_answers", required=True)
    ap.add_argument("--out_csv", default="analysis/exp_c/faithfulness_sheet.csv")
    ap.add_argument("--sample_size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--bm25_dir", default="indexes/bm25_v3")
    ap.add_argument("--dense_dir", default="indexes/dense_gte-Qwen2-7B-instruct_v2")
    ap.add_argument("--embed_model", default="Alibaba-NLP/gte-Qwen2-7B-instruct")
    ap.add_argument("--embed_quant_backend", choices=["none", "8bit", "4bit"], default="8bit")
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--w_dense", type=float, default=0.5)
    ap.add_argument("--w_sparse", type=float, default=0.5)
    ap.add_argument("--k_dense", type=int, default=200)
    ap.add_argument("--k_sparse", type=int, default=200)
    ap.add_argument("--rrf_k", type=int, default=60)

    ap.add_argument("--sparse_title_weight", type=int, default=3)
    ap.add_argument("--sparse_heading_weight", type=int, default=2)
    ap.add_argument("--sparse_body_weight", type=int, default=1)
    ap.add_argument("--sparse_add_bigrams", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sparse_prf", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sparse_prf_k", type=int, default=8)
    ap.add_argument("--sparse_prf_terms", type=int, default=6)
    ap.add_argument("--sparse_prf_alpha", type=float, default=0.65)

    ap.add_argument("--k_retrieve", type=int, default=100)
    ap.add_argument("--stage1_k", type=int, default=60)
    ap.add_argument("--mmr_lambda", type=float, default=0.75)
    ap.add_argument("--temporal_boost_weight", type=float, default=0.12)
    ap.add_argument("--top_ctx", type=int, default=3, help="Number of context chunks to expose for manual check")
    args = ap.parse_args()

    queries = load_queries(Path(args.queries))
    chunk_map = load_chunk_text_map(Path(args.chunks))
    closed_answers = load_answers(Path(args.closed_answers))
    rag_answers = load_answers(Path(args.rag_answers))

    q_by_id = {x["id"]: x for x in queries}
    qids_by_type: Dict[str, List[str]] = defaultdict(list)
    for item in queries:
        qids_by_type[classify_question_type(item["question"])].append(item["id"])

    sampled_ids = stratified_sample(qids_by_type, sample_size=args.sample_size, seed=args.seed)

    retriever = build_retriever(
        mode="hybrid",
        bm25_dir=Path(args.bm25_dir),
        dense_dir=Path(args.dense_dir),
        chunks_path=Path(args.chunks),
        w_dense=args.w_dense,
        w_sparse=args.w_sparse,
        k_dense=args.k_dense,
        k_sparse=args.k_sparse,
        device=args.device,
        model_name=args.embed_model,
        quant_backend=args.embed_quant_backend,
        sparse_title_weight=args.sparse_title_weight,
        sparse_heading_weight=args.sparse_heading_weight,
        sparse_body_weight=args.sparse_body_weight,
        sparse_add_bigrams=args.sparse_add_bigrams,
        sparse_prf=args.sparse_prf,
        sparse_prf_k=args.sparse_prf_k,
        sparse_prf_terms=args.sparse_prf_terms,
        sparse_prf_alpha=args.sparse_prf_alpha,
        fusion_method="rrf",
        rrf_k=args.rrf_k,
    )

    rows: List[Dict[str, Any]] = []
    for qid in sampled_ids:
        item = q_by_id[qid]
        q = item["question"]
        q_type = classify_question_type(q)
        retrieved = retriever.retrieve(q, k=args.k_retrieve)
        selected_ids = mmr_select_chunk_ids(
            query=q,
            retrieved=retrieved,
            chunk_map=chunk_map,
            retriever=retriever,
            stage1_k=args.stage1_k,
            out_k=max(1, args.top_ctx),
            mmr_lambda=args.mmr_lambda,
            temporal_boost_weight=args.temporal_boost_weight,
            use_mmr=True,
        )
        if not selected_ids:
            selected_ids = [cid for cid, _ in retrieved[: args.top_ctx] if cid in chunk_map]

        row: Dict[str, Any] = {
            "qid": qid,
            "question_type": q_type,
            "question": q,
            "closed_answer": _answer_for(qid, closed_answers),
            "rag_answer": _answer_for(qid, rag_answers),
            # Fill these manually with: supported / unsupported / idk
            "closed_label": "",
            "rag_label": "",
            "annotation_notes": "",
        }
        for i in range(args.top_ctx):
            prefix = f"ctx{i+1}"
            cid = selected_ids[i] if i < len(selected_ids) else ""
            c = chunk_map.get(cid, {}) if cid else {}
            row[f"{prefix}_chunk_id"] = cid
            row[f"{prefix}_title"] = str(c.get("title", "") or "")
            row[f"{prefix}_domain"] = str(c.get("domain", "") or "")
            row[f"{prefix}_source_url"] = str(c.get("source_url", "") or "")
            row[f"{prefix}_snippet"] = snippet(str(c.get("text", "") or ""), 220)
        rows.append(row)

    fieldnames = [
        "qid",
        "question_type",
        "question",
        "closed_answer",
        "rag_answer",
        "closed_label",
        "rag_label",
        "annotation_notes",
    ]
    for i in range(args.top_ctx):
        prefix = f"ctx{i+1}"
        fieldnames.extend(
            [
                f"{prefix}_chunk_id",
                f"{prefix}_title",
                f"{prefix}_domain",
                f"{prefix}_source_url",
                f"{prefix}_snippet",
            ]
        )

    out_csv = Path(args.out_csv)
    write_csv(out_csv, rows, fieldnames)
    print(f"[exp_c_prepare] wrote {out_csv} with {len(rows)} sampled queries")


if __name__ == "__main__":
    main()

