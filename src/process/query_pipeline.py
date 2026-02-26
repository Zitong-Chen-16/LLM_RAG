#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from hybrid_retrieval import build_default_hybrid
from reader import QwenReader, ReaderConfig
from utils import load_chunk_text_map
from dense_retrieval import build_embed_text
import numpy as np

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


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def mmr_select_chunk_ids(
    *,
    query: str,
    retrieved: List[tuple[str, float]],
    chunk_map: Dict[str, Dict[str, Any]],
    retriever,
    stage1_k: int,
    out_k: int,
    mmr_lambda: float,
) -> List[str]:
    cands = [(cid, float(sc)) for cid, sc in retrieved if cid in chunk_map][:stage1_k]
    if not cands:
        return []

    cand_ids = [cid for cid, _ in cands]
    fused_scores = np.array([sc for _cid, sc in cands], dtype=np.float32)
    fused_n = _minmax(fused_scores)

    dense = retriever.dense
    q_emb = dense.encode_texts([query], batch_size=1)[0]
    doc_texts = [build_embed_text(chunk_map[cid]) for cid in cand_ids]
    doc_emb = dense.encode_texts(doc_texts, batch_size=min(len(doc_texts), 64))
    dense_rel = (doc_emb @ q_emb).astype(np.float32)
    dense_rel_n = _minmax(dense_rel)

    # Blend dense relevance with hybrid prior, then enforce diversity with MMR.
    rel = (0.8 * dense_rel_n + 0.2 * fused_n).astype(np.float32)

    selected: List[int] = []
    remaining = set(range(len(cand_ids)))
    while remaining and len(selected) < out_k:
        if not selected:
            best = max(remaining, key=lambda i: float(rel[i]))
            selected.append(best)
            remaining.remove(best)
            continue

        best = None
        best_score = None
        for i in remaining:
            max_sim = max(float(doc_emb[i] @ doc_emb[j]) for j in selected)
            score = float(mmr_lambda * rel[i] - (1.0 - mmr_lambda) * max_sim)
            if best is None or score > best_score:
                best = i
                best_score = score

        selected.append(best)
        remaining.remove(best)

    return [cand_ids[i] for i in selected]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="leaderboard_queries.json")
    ap.add_argument("--out", default="leaderboard_answers.json")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--bm25_dir", default="indexes/bm25")
    ap.add_argument("--dense_dir", default="indexes/dense_gte-Qwen2-1.5B-instruct")

    ap.add_argument("--k_retrieve", type=int, default=20, help="Initial candidate count from hybrid retriever")
    ap.add_argument("--k_ctx", type=int, default=20, help="How many chunks to pass to reader")
    # ap.add_argument("--stage1_k", type=int, default=20, help="Top candidates considered by MMR reranker")
    # ap.add_argument("--mmr_lambda", type=float, default=0.75, help="MMR relevance/diversity weight")
    # ap.add_argument("--dedup_doc", action="store_true", help="Deduplicate context chunks by doc_id")

    ap.add_argument("--w_dense", type=float, default=0.6)
    ap.add_argument("--w_sparse", type=float, default=0.4)
    ap.add_argument("--k_dense", type=int, default=150)
    ap.add_argument("--k_sparse", type=int, default=150)
    ap.add_argument("--fusion_method", choices=["rrf", "minmax"], default="rrf")
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--embed_model", default="Alibaba-NLP/gte-Qwen2-1.5B-instruct")

    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--reader_device", type=str, default="cuda:0", help="Reader device: auto / cuda:0 / cuda:1")
    ap.add_argument("--quant_backend", choices=["auto", "bnb", "gptq", "none"], default="auto")
    ap.add_argument("--max_context_tokens", type=int, default=12000)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
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
        model_name=args.embed_model,
        fusion_method=args.fusion_method,
        rrf_k=args.rrf_k,
    )

    reader = QwenReader(ReaderConfig(
        model_name=args.model,
        device_map=args.reader_device if args.reader_device == "auto" else {"": args.reader_device},
        quant_backend=args.quant_backend,
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
        # selected_ids = mmr_select_chunk_ids(
        #     query=q,
        #     retrieved=retrieved,
        #     chunk_map=chunk_map,
        #     retriever=retriever,
        #     stage1_k=args.stage1_k,
        #     out_k=args.k_ctx,
        #     mmr_lambda=args.mmr_lambda,
        # )
        # ctx = [chunk_map[cid] for cid in selected_ids if cid in chunk_map]
        ctx = [chunk_map[cid] for cid, _ in retrieved if cid in chunk_map]
        # if len(ctx) < args.k_ctx:
        #     seen_ids = set(selected_ids)
        #     for cid, _sc in retrieved:
        #         if cid in chunk_map and cid not in seen_ids:
        #             ctx.append(chunk_map[cid])
        #             seen_ids.add(cid)
        #             if len(ctx) >= args.k_ctx:
        #                 break

        # if args.dedup_doc:
        #     ctx = dedup_by_doc_id(ctx, args.k_ctx)
        # else:
        #     ctx = ctx[:args.k_ctx]

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
