#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS = Path(__file__).resolve()
if str(THIS.parent) not in sys.path:
    sys.path.insert(0, str(THIS.parent))

from common import (  # noqa: E402
    classify_question_type,
    load_answers,
    load_queries,
    normalize_label,
    write_json,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_idk(ans: str) -> bool:
    s = (ans or "").strip().lower()
    return s in {"", "i don't know", "i dont know", "unknown", "not sure"}


def _load_label_map(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        qid = str(r.get("qid", "")).strip()
        if not qid:
            continue
        out[qid] = {
            "closed": normalize_label(str(r.get("closed_label", "") or "")),
            "rag": normalize_label(str(r.get("rag_label", "") or "")),
        }
    return out


def _top_titles_snips(detail: Optional[Dict[str, Any]]) -> List[tuple[str, str]]:
    if not detail:
        return []
    titles = detail.get("top_titles") or []
    snips = detail.get("top_snippets") or []
    out = []
    n = min(len(titles), len(snips), 3)
    for i in range(n):
        out.append((str(titles[i] or ""), str(snips[i] or "")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Experiment D representative examples")
    ap.add_argument("--queries", default="leaderboard_queries.json")
    ap.add_argument("--details_jsonl", default="analysis/exp_ab/retrieval_details.jsonl")
    ap.add_argument("--closed_answers", required=True)
    ap.add_argument("--dense_answers", required=True)
    ap.add_argument("--sparse_answers", required=True)
    ap.add_argument("--hybrid_answers", required=True)
    ap.add_argument("--hybrid_minmax_answers", default="", help="Optional")
    ap.add_argument("--faithfulness_csv", default="", help="Optional annotated sheet from Experiment C")
    ap.add_argument("--out_md", default="analysis/exp_d/representative_examples.md")
    ap.add_argument("--out_json", default="analysis/exp_d/selected_examples.json")
    ap.add_argument("--n_per_bucket", type=int, default=2)
    args = ap.parse_args()

    queries = load_queries(Path(args.queries))
    q_by_id = {x["id"]: x["question"] for x in queries}
    qtype_by_id = {x["id"]: classify_question_type(x["question"]) for x in queries}

    closed = load_answers(Path(args.closed_answers))
    dense = load_answers(Path(args.dense_answers))
    sparse = load_answers(Path(args.sparse_answers))
    hybrid = load_answers(Path(args.hybrid_answers))
    minmax = load_answers(Path(args.hybrid_minmax_answers)) if args.hybrid_minmax_answers else {}
    labels = _load_label_map(Path(args.faithfulness_csv)) if args.faithfulness_csv else {}

    details_rows = _read_jsonl(Path(args.details_jsonl))
    detail_by_method_qid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in details_rows:
        m = str(r.get("method", "")).strip()
        qid = str(r.get("qid", "")).strip()
        if not m or not qid:
            continue
        detail_by_method_qid.setdefault(m, {})[qid] = r

    def d(method: str, qid: str) -> Optional[Dict[str, Any]]:
        return detail_by_method_qid.get(method, {}).get(qid)

    # Candidate buckets.
    cand_hybrid_over_dense: List[str] = []
    cand_hybrid_over_sparse: List[str] = []
    cand_closed_hallu_rag_supported: List[str] = []
    cand_hybrid_fail: List[str] = []

    for qid in q_by_id:
        dense_d = d("dense", qid)
        sparse_d = d("sparse", qid)
        hyr_d = d("hybrid_rrf", qid)
        if not hyr_d:
            continue

        hyr_hit = int(hyr_d.get("evidence_proxy_hit", 0))
        dense_hit = int((dense_d or {}).get("evidence_proxy_hit", 0))
        sparse_hit = int((sparse_d or {}).get("evidence_proxy_hit", 0))

        if hyr_hit == 1 and dense_hit == 0:
            cand_hybrid_over_dense.append(qid)
        if hyr_hit == 1 and sparse_hit == 0:
            cand_hybrid_over_sparse.append(qid)

        # Prefer strict manual labels when available.
        if qid in labels and labels[qid]["closed"] and labels[qid]["rag"]:
            if labels[qid]["closed"] == "unsupported" and labels[qid]["rag"] == "supported":
                cand_closed_hallu_rag_supported.append(qid)
        else:
            if (
                hyr_hit == 1
                and not _is_idk(closed.get(qid, ""))
                and not _is_idk(hybrid.get(qid, ""))
                and closed.get(qid, "").strip() != hybrid.get(qid, "").strip()
            ):
                cand_closed_hallu_rag_supported.append(qid)

        if hyr_hit == 0 or _is_idk(hybrid.get(qid, "")):
            cand_hybrid_fail.append(qid)

    # Deterministic selection.
    cand_hybrid_over_dense = sorted(set(cand_hybrid_over_dense))
    cand_hybrid_over_sparse = sorted(set(cand_hybrid_over_sparse))
    cand_closed_hallu_rag_supported = sorted(set(cand_closed_hallu_rag_supported))
    cand_hybrid_fail = sorted(set(cand_hybrid_fail))

    selected = {
        "hybrid_wins_dense": cand_hybrid_over_dense[: args.n_per_bucket],
        "hybrid_wins_sparse": cand_hybrid_over_sparse[: args.n_per_bucket],
        "closed_hallucinates_rag_supported": cand_closed_hallu_rag_supported[: args.n_per_bucket],
        "hybrid_failure_cases": cand_hybrid_fail[: args.n_per_bucket],
    }
    write_json(Path(args.out_json), selected)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Representative Examples (Experiment D)")
    lines.append("")
    lines.append(
        "Selection buckets: 2 hybrid>dense, 2 hybrid>sparse, 2 closed-hallucination vs RAG-supported, 2 failures."
    )
    lines.append("")

    bucket_titles = {
        "hybrid_wins_dense": "A) Hybrid (RRF) beats Dense-only",
        "hybrid_wins_sparse": "B) Hybrid (RRF) beats Sparse-only",
        "closed_hallucinates_rag_supported": "C) Closed-book hallucination vs RAG-supported",
        "hybrid_failure_cases": "D) Hybrid failure cases",
    }

    for key in [
        "hybrid_wins_dense",
        "hybrid_wins_sparse",
        "closed_hallucinates_rag_supported",
        "hybrid_failure_cases",
    ]:
        lines.append(f"## {bucket_titles[key]}")
        lines.append("")
        qids = selected.get(key, [])
        if not qids:
            lines.append("_No candidates found with current heuristics/labels._")
            lines.append("")
            continue

        for qid in qids:
            q = q_by_id.get(qid, "")
            qtype = qtype_by_id.get(qid, "")
            lines.append(f"### QID {qid} ({qtype})")
            lines.append(f"**Question:** {q}")
            lines.append("")
            lines.append(f"- Closed-book: {closed.get(qid, '')}")
            lines.append(f"- Dense-only: {dense.get(qid, '')}")
            lines.append(f"- Sparse-only: {sparse.get(qid, '')}")
            lines.append(f"- Hybrid RRF: {hybrid.get(qid, '')}")
            if minmax:
                lines.append(f"- Hybrid MinMax: {minmax.get(qid, '')}")
            if qid in labels:
                lines.append(
                    f"- Manual labels: closed={labels[qid].get('closed','')}, rag={labels[qid].get('rag','')}"
                )
            lines.append("")

            for method, label in [
                ("hybrid_rrf", "Hybrid RRF top retrieval"),
                ("dense", "Dense-only top retrieval"),
                ("sparse", "Sparse-only top retrieval"),
            ]:
                pairs = _top_titles_snips(d(method, qid))
                lines.append(f"**{label}:**")
                if not pairs:
                    lines.append("- _No retrieval detail found_")
                else:
                    for i, (title, snip) in enumerate(pairs, start=1):
                        lines.append(f"- {i}. {title} | {snip}")
                lines.append("")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[exp_d] wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
