#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

THIS = Path(__file__).resolve()
if str(THIS.parent) not in sys.path:
    sys.path.insert(0, str(THIS.parent))

from common import normalize_label, rate, write_json  # noqa: E402


def _summarize_labels(labels: List[str]) -> Dict[str, float]:
    n = len(labels)
    supported = sum(1 for x in labels if x == "supported")
    unsupported = sum(1 for x in labels if x == "unsupported")
    idk = sum(1 for x in labels if x == "idk")
    supported_or_unsupported = supported + unsupported
    return {
        "n_labeled": n,
        "supported": supported,
        "unsupported": unsupported,
        "idk": idk,
        "support_rate": round(rate(supported, n), 4),
        "idk_rate": round(rate(idk, n), 4),
        # Hallucination proxy: unsupported among non-IDK responses.
        "hallucination_rate": round(rate(unsupported, supported_or_unsupported), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize manual Experiment C faithfulness labels")
    ap.add_argument("--in_csv", default="analysis/exp_c/faithfulness_sheet.csv")
    ap.add_argument("--out_json", default="analysis/exp_c/faithfulness_summary.json")
    args = ap.parse_args()

    path = Path(args.in_csv)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    closed_labels: List[str] = []
    rag_labels: List[str] = []
    by_type: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"closed": [], "rag": []})

    for r in rows:
        q_type = str(r.get("question_type", "") or "").strip() or "unknown"
        c = normalize_label(str(r.get("closed_label", "") or ""))
        g = normalize_label(str(r.get("rag_label", "") or ""))
        if c:
            closed_labels.append(c)
            by_type[q_type]["closed"].append(c)
        if g:
            rag_labels.append(g)
            by_type[q_type]["rag"].append(g)

    overall = {
        "closed_book": _summarize_labels(closed_labels),
        "rag_hybrid_rrf": _summarize_labels(rag_labels),
    }
    overall["delta"] = {
        "support_rate_gain": round(
            overall["rag_hybrid_rrf"]["support_rate"] - overall["closed_book"]["support_rate"], 4
        ),
        "hallucination_rate_reduction": round(
            overall["closed_book"]["hallucination_rate"] - overall["rag_hybrid_rrf"]["hallucination_rate"], 4
        ),
    }

    by_type_summary = {}
    for q_type, d in sorted(by_type.items()):
        by_type_summary[q_type] = {
            "closed_book": _summarize_labels(d["closed"]),
            "rag_hybrid_rrf": _summarize_labels(d["rag"]),
        }
        by_type_summary[q_type]["delta"] = {
            "support_rate_gain": round(
                by_type_summary[q_type]["rag_hybrid_rrf"]["support_rate"]
                - by_type_summary[q_type]["closed_book"]["support_rate"],
                4,
            ),
            "hallucination_rate_reduction": round(
                by_type_summary[q_type]["closed_book"]["hallucination_rate"]
                - by_type_summary[q_type]["rag_hybrid_rrf"]["hallucination_rate"],
                4,
            ),
        }

    out = {
        "overall": overall,
        "by_question_type": by_type_summary,
        "notes": (
            "hallucination_rate is computed as unsupported / (supported + unsupported), "
            "excluding IDK from denominator."
        ),
    }
    write_json(Path(args.out_json), out)

    print(f"[exp_c_summary] wrote {args.out_json}")
    print("[exp_c_summary] overall:", overall)


if __name__ == "__main__":
    main()

