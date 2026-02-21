#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from hybrid_retrieval import build_default_hybrid
from reader import QwenReader, ReaderConfig
from utils import load_chunk_text_map

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dedup_by_doc_id(chunks: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for c in chunks:
        doc_id = c.get("doc_id")
        if doc_id and doc_id in seen:
            continue
        if doc_id:
            seen.add(doc_id)
        out.append(c)
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="leaderboard_queries.json")
    ap.add_argument("--out", default="leaderboard_answers.json")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--bm25_dir", default="indexes/bm25")
    ap.add_argument("--dense_dir", default="indexes/dense")

    ap.add_argument("--k_retrieve", type=int, default=10, help="Final k returned from hybrid retriever")
    ap.add_argument("--k_ctx", type=int, default=10, help="How many chunks to pass to reader")
    ap.add_argument("--dedup_doc", action="store_true", help="Deduplicate context chunks by doc_id")

    ap.add_argument("--w_dense", type=float, default=0.6)
    ap.add_argument("--w_sparse", type=float, default=0.4)
    ap.add_argument("--k_dense", type=int, default=100)
    ap.add_argument("--k_sparse", type=int, default=100)

    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--max_context_tokens", type=int, default=12000)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--null_answer", default="I don't know")
    ap.add_argument("--device", type=str, default="cuda:1")
    args = ap.parse_args()

    queries_path = Path(args.queries)
    out_path = Path(args.out)
    chunks_path = Path(args.chunks)

    queries = load_json(queries_path)
    if not isinstance(queries, list):
        raise ValueError("Expected queries JSON to be a list of objects with keys {id, question}.")

    chunk_map = load_chunk_text_map(chunks_path)

    retriever = build_default_hybrid(
        bm25_dir=Path(args.bm25_dir),
        dense_dir=Path(args.dense_dir),
        chunks_path=chunks_path,
        w_dense=args.w_dense,
        w_sparse=args.w_sparse,
        k_dense=args.k_dense,
        k_sparse=args.k_sparse,
        device=args.device,
    )

    reader = QwenReader(ReaderConfig(
        model_name=args.model,
        load_in_4bit=True,
        max_context_tokens=args.max_context_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ))
    reader.load()

    out: Dict[str, str] = {}

    for item in queries:
        qid = str(item.get("id", "")).strip()
        q = str(item.get("question", "")).strip()
        if not qid or not q:
            continue

        retrieved = retriever.retrieve(q, k=args.k_retrieve)
        ctx = [chunk_map[cid] for cid, _ in retrieved if cid in chunk_map]
        ctx = ctx[: max(args.k_ctx * 3, args.k_ctx)]

        if args.dedup_doc:
            ctx = dedup_by_doc_id(ctx, args.k_ctx)
        else:
            ctx = ctx[:args.k_ctx]

        ans, _used = reader.answer(q, ctx)
        ans = (ans or "").strip()
        if not ans:
            ans = args.null_answer

        out[qid] = ans

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with {len(out)} answers.")


if __name__ == "__main__":
    main()
