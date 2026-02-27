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
    evidence_proxy_hit,
    load_queries,
    mean,
    snippet,
    write_csv,
    write_json,
    write_jsonl,
)
from hybrid_retrieval import build_retriever  # noqa: E402
from utils import load_chunk_text_map  # noqa: E402


def _build_method_retriever(method: str, args, chunks_path: Path):
    method = method.strip().lower()
    if method == "dense":
        return build_retriever(
            mode="dense",
            dense_dir=Path(args.dense_dir),
            chunks_path=chunks_path,
            k_dense=args.k_dense,
            device=args.device,
            model_name=args.embed_model,
            quant_backend=args.embed_quant_backend,
        )
    if method == "sparse":
        return build_retriever(
            mode="sparse",
            bm25_dir=Path(args.bm25_dir),
            chunks_path=chunks_path,
            k_sparse=args.k_sparse,
            sparse_title_weight=args.sparse_title_weight,
            sparse_heading_weight=args.sparse_heading_weight,
            sparse_body_weight=args.sparse_body_weight,
            sparse_add_bigrams=args.sparse_add_bigrams,
            sparse_prf=args.sparse_prf,
            sparse_prf_k=args.sparse_prf_k,
            sparse_prf_terms=args.sparse_prf_terms,
            sparse_prf_alpha=args.sparse_prf_alpha,
        )
    if method == "hybrid_rrf":
        return build_retriever(
            mode="hybrid",
            bm25_dir=Path(args.bm25_dir),
            dense_dir=Path(args.dense_dir),
            chunks_path=chunks_path,
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
    if method == "hybrid_minmax":
        return build_retriever(
            mode="hybrid",
            bm25_dir=Path(args.bm25_dir),
            dense_dir=Path(args.dense_dir),
            chunks_path=chunks_path,
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
            fusion_method="minmax",
            rrf_k=args.rrf_k,
        )
    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment A/B retrieval diagnostics")
    ap.add_argument("--queries", default="leaderboard_queries.json")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--out_dir", default="analysis/exp_ab")
    ap.add_argument("--methods", default="sparse,dense,hybrid_rrf,hybrid_minmax")
    ap.add_argument("--k", type=int, default=10, help="Top-k retrieved chunks for diagnostics")

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
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(Path(args.queries))
    chunk_map = load_chunk_text_map(Path(args.chunks))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # Experiment A: type counts
    type_counts: Dict[str, int] = defaultdict(int)
    qtype_by_qid: Dict[str, str] = {}
    for item in queries:
        qtype = classify_question_type(item["question"])
        qtype_by_qid[item["id"]] = qtype
        type_counts[qtype] += 1

    write_json(out_dir / "query_type_counts.json", dict(sorted(type_counts.items())))
    type_rows = [{"question_type": t, "count": c} for t, c in sorted(type_counts.items())]
    write_csv(out_dir / "query_type_counts.csv", type_rows, ["question_type", "count"])

    # Build retrievers once.
    retrievers = {m: _build_method_retriever(m, args, Path(args.chunks)) for m in methods}

    details: List[Dict[str, Any]] = []
    acc_method: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    acc_method_type: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for item in queries:
        qid = item["id"]
        q = item["question"]
        q_type = qtype_by_qid[qid]

        for method in methods:
            retrieved = retrievers[method].retrieve(q, k=args.k)
            top = [(cid, float(sc)) for cid, sc in retrieved[: args.k] if cid in chunk_map]
            chunks = [chunk_map[cid] for cid, _ in top]
            doc_ids = [str(c.get("doc_id", "")) for c in chunks if c.get("doc_id")]
            domains = [str(c.get("domain", "")) for c in chunks if c.get("domain")]
            doc_div = float(len(set(doc_ids)))
            dom_div = float(len(set(domains)))
            proxy_hit = 1.0 if evidence_proxy_hit(q_type, q, chunks) else 0.0

            top_titles = [str(c.get("title", "") or "") for c in chunks[:3]]
            top_snips = [snippet(str(c.get("text", "") or ""), 160) for c in chunks[:3]]

            row = {
                "qid": qid,
                "question": q,
                "question_type": q_type,
                "method": method,
                "k": args.k,
                "doc_diversity": doc_div,
                "domain_diversity": dom_div,
                "evidence_proxy_hit": int(proxy_hit),
                "top_chunk_ids": [cid for cid, _ in top],
                "top_doc_ids": doc_ids,
                "top_domains": domains,
                "top_titles": top_titles,
                "top_snippets": top_snips,
            }
            details.append(row)

            acc_method[method]["doc_diversity"].append(doc_div)
            acc_method[method]["domain_diversity"].append(dom_div)
            acc_method[method]["evidence_proxy_hit"].append(proxy_hit)

            key = (method, q_type)
            acc_method_type[key]["doc_diversity"].append(doc_div)
            acc_method_type[key]["domain_diversity"].append(dom_div)
            acc_method_type[key]["evidence_proxy_hit"].append(proxy_hit)

    write_jsonl(out_dir / "retrieval_details.jsonl", details)

    summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        m = acc_method[method]
        summary_rows.append(
            {
                "method": method,
                "n_queries": len(m["doc_diversity"]),
                "doc_diversity_mean": round(mean(m["doc_diversity"]), 4),
                "domain_diversity_mean": round(mean(m["domain_diversity"]), 4),
                "evidence_proxy_rate": round(mean(m["evidence_proxy_hit"]), 4),
            }
        )
    write_csv(
        out_dir / "summary_by_method.csv",
        summary_rows,
        ["method", "n_queries", "doc_diversity_mean", "domain_diversity_mean", "evidence_proxy_rate"],
    )
    write_json(out_dir / "summary_by_method.json", summary_rows)

    summary_type_rows: List[Dict[str, Any]] = []
    for (method, q_type), m in sorted(acc_method_type.items()):
        summary_type_rows.append(
            {
                "method": method,
                "question_type": q_type,
                "n_queries": len(m["doc_diversity"]),
                "doc_diversity_mean": round(mean(m["doc_diversity"]), 4),
                "domain_diversity_mean": round(mean(m["domain_diversity"]), 4),
                "evidence_proxy_rate": round(mean(m["evidence_proxy_hit"]), 4),
            }
        )
    write_csv(
        out_dir / "summary_by_method_type.csv",
        summary_type_rows,
        [
            "method",
            "question_type",
            "n_queries",
            "doc_diversity_mean",
            "domain_diversity_mean",
            "evidence_proxy_rate",
        ],
    )
    write_json(out_dir / "summary_by_method_type.json", summary_type_rows)

    print(f"[exp_ab] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()

