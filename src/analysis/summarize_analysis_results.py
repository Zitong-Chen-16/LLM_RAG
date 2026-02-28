#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_markdown_table(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_label(method: str, source: str = "main") -> str:
    m = (method or "").strip().lower()
    if m == "sparse":
        return "Sparse-only"
    if m == "dense":
        return "Dense-only"
    if m == "hybrid_rrf":
        return "Hybrid (RRF, MiniLM)" if source == "exp8" else "Hybrid (RRF, GTE-Qwen)"
    if m == "hybrid_minmax":
        return "Hybrid (MinMax, GTE-Qwen)"
    return method


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _load_queries(path: Path) -> Dict[str, Dict[str, str]]:
    data = _read_json(path) or []
    out: Dict[str, Dict[str, str]] = {}
    if isinstance(data, list):
        for item in data:
            qid = str(item.get("id", "")).strip()
            q = str(item.get("question", "")).strip()
            if qid:
                out[qid] = {"question": q}
    return out


def _load_type_map(path: Path) -> Dict[str, str]:
    rows = _read_csv(path)
    out: Dict[str, str] = {}
    for r in rows:
        out[str(r.get("qid", "")).strip()] = str(r.get("question_type", "")).strip()
    return out


def _build_example_rows(
    selected_examples_path: Path,
    queries_path: Path,
    type_map_main: Dict[str, str],
) -> List[Dict[str, Any]]:
    selected = _read_json(selected_examples_path) or {}
    qmap = _load_queries(queries_path)
    rows: List[Dict[str, Any]] = []
    bucket_titles = {
        "hybrid_wins_dense": "Hybrid > Dense",
        "hybrid_wins_sparse": "Hybrid > Sparse",
        "hybrid_qwen_vs_hybrid_minilm": "Hybrid GTE-Qwen vs Hybrid MiniLM",
        "hybrid_failure_cases": "Hybrid Failure",
    }
    for bucket, qids in selected.items():
        if not isinstance(qids, list):
            continue
        for qid in qids:
            qid_s = str(qid)
            rows.append(
                {
                    "bucket": bucket_titles.get(bucket, bucket),
                    "qid": qid_s,
                    "question_type": type_map_main.get(qid_s, ""),
                    "question": qmap.get(qid_s, {}).get("question", ""),
                }
            )
    return rows


def _try_plot_overall(
    out_dir: Path,
    overall_rows: List[Dict[str, Any]],
    by_type_rows: List[Dict[str, Any]],
    hybrid_embed_rows: List[Dict[str, Any]],
    type_order: List[str],
    method_order: List[str],
) -> List[str]:
    generated: List[str] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return _fallback_svg_plots(
            out_dir=out_dir,
            overall_rows=overall_rows,
            by_type_rows=by_type_rows,
            hybrid_embed_rows=hybrid_embed_rows,
            type_order=type_order,
            method_order=method_order,
        )

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Evidence proxy rate by method
    methods = [r["method"] for r in overall_rows]
    evidence = [_to_float(r["evidence_proxy_rate"]) for r in overall_rows]
    plt.figure(figsize=(10, 4.5))
    plt.bar(methods, evidence)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Evidence Proxy Rate")
    plt.title("Retrieval Evidence Proxy Rate by Method")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    p1 = fig_dir / "overall_evidence_proxy_rate.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    generated.append(str(p1))

    # 2) Diversity metrics by method
    doc_div = [_to_float(r["doc_diversity_mean"]) for r in overall_rows]
    dom_div = [_to_float(r["domain_diversity_mean"]) for r in overall_rows]
    x = list(range(len(methods)))
    width = 0.38
    plt.figure(figsize=(10, 4.5))
    plt.bar([i - width / 2 for i in x], doc_div, width=width, label="Doc Diversity")
    plt.bar([i + width / 2 for i in x], dom_div, width=width, label="Domain Diversity")
    plt.xticks(x, methods, rotation=20, ha="right")
    plt.ylabel("Top-k Diversity (Mean)")
    plt.title("Retrieval Diversity by Method")
    plt.legend()
    plt.tight_layout()
    p2 = fig_dir / "overall_diversity.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    generated.append(str(p2))

    # 3) Evidence proxy heatmap by method x question type
    pivot: Dict[Tuple[str, str], float] = {}
    for r in by_type_rows:
        pivot[(r["method"], r["question_type"])] = _to_float(r["evidence_proxy_rate"])

    data = [
        [pivot.get((m, t), 0.0) for t in type_order]
        for m in method_order
    ]
    plt.figure(figsize=(1.8 * max(3, len(type_order)), 0.7 * max(3, len(method_order)) + 2.2))
    im = plt.imshow(data, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Evidence Proxy Rate")
    plt.xticks(range(len(type_order)), type_order, rotation=20, ha="right")
    plt.yticks(range(len(method_order)), method_order)
    plt.title("Evidence Proxy Rate by Question Type and Method")
    for i in range(len(method_order)):
        for j in range(len(type_order)):
            plt.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    p3 = fig_dir / "evidence_proxy_heatmap.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    generated.append(str(p3))

    # 4) Doc diversity heatmap by method x question type
    pivot_doc: Dict[Tuple[str, str], float] = {}
    for r in by_type_rows:
        pivot_doc[(r["method"], r["question_type"])] = _to_float(r["doc_diversity_mean"])
    data_doc = [[pivot_doc.get((m, t), 0.0) for t in type_order] for m in method_order]
    vmax_doc = max([max(row) for row in data_doc], default=1.0)
    if vmax_doc <= 0:
        vmax_doc = 1.0
    plt.figure(figsize=(1.8 * max(3, len(type_order)), 0.7 * max(3, len(method_order)) + 2.2))
    im = plt.imshow(data_doc, aspect="auto", vmin=0.0, vmax=vmax_doc)
    plt.colorbar(im, label="Doc Diversity Mean")
    plt.xticks(range(len(type_order)), type_order, rotation=20, ha="right")
    plt.yticks(range(len(method_order)), method_order)
    plt.title("Doc Diversity Mean by Question Type and Method")
    for i in range(len(method_order)):
        for j in range(len(type_order)):
            plt.text(j, i, f"{data_doc[i][j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    p4 = fig_dir / "doc_diversity_heatmap.png"
    plt.savefig(p4, dpi=150)
    plt.close()
    generated.append(str(p4))

    # 5) Domain diversity heatmap by method x question type
    pivot_dom: Dict[Tuple[str, str], float] = {}
    for r in by_type_rows:
        pivot_dom[(r["method"], r["question_type"])] = _to_float(r["domain_diversity_mean"])
    data_dom = [[pivot_dom.get((m, t), 0.0) for t in type_order] for m in method_order]
    vmax_dom = max([max(row) for row in data_dom], default=1.0)
    if vmax_dom <= 0:
        vmax_dom = 1.0
    plt.figure(figsize=(1.8 * max(3, len(type_order)), 0.7 * max(3, len(method_order)) + 2.2))
    im = plt.imshow(data_dom, aspect="auto", vmin=0.0, vmax=vmax_dom)
    plt.colorbar(im, label="Domain Diversity Mean")
    plt.xticks(range(len(type_order)), type_order, rotation=20, ha="right")
    plt.yticks(range(len(method_order)), method_order)
    plt.title("Domain Diversity Mean by Question Type and Method")
    for i in range(len(method_order)):
        for j in range(len(type_order)):
            plt.text(j, i, f"{data_dom[i][j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    p5 = fig_dir / "domain_diversity_heatmap.png"
    plt.savefig(p5, dpi=150)
    plt.close()
    generated.append(str(p5))

    # 6) Hybrid embedder comparison (if both rows exist)
    if len(hybrid_embed_rows) >= 2:
        labels = [r["method"] for r in hybrid_embed_rows]
        metrics = ["doc_diversity_mean", "domain_diversity_mean", "evidence_proxy_rate"]
        x = list(range(len(metrics)))
        width = 0.36
        plt.figure(figsize=(8, 4.5))
        a = [_to_float(hybrid_embed_rows[0][m]) for m in metrics]
        b = [_to_float(hybrid_embed_rows[1][m]) for m in metrics]
        plt.bar([i - width / 2 for i in x], a, width=width, label=labels[0])
        plt.bar([i + width / 2 for i in x], b, width=width, label=labels[1])
        plt.xticks(x, ["Doc Diversity", "Domain Diversity", "Evidence Proxy"])
        plt.title("Hybrid Embedder Ablation (GTE-Qwen vs MiniLM)")
        plt.legend()
        plt.tight_layout()
        p6 = fig_dir / "hybrid_embedder_ablation.png"
        plt.savefig(p6, dpi=150)
        plt.close()
        generated.append(str(p6))

    return generated


def _fallback_svg_bar(
    path: Path,
    title: str,
    categories: List[str],
    series: List[Tuple[str, List[float]]],
    y_max: Optional[float] = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(categories)
    if n == 0 or not series:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='120'></svg>", encoding="utf-8")
        return str(path)

    width = max(720, 120 + n * 110)
    height = 420
    margin_left = 70
    margin_right = 20
    margin_top = 45
    margin_bottom = 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    vmax = y_max if y_max is not None else max(max(vals) for _name, vals in series)
    if vmax <= 0:
        vmax = 1.0
    vmax *= 1.1

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#9c755f", "#edc949"]
    s_count = len(series)
    group_w = plot_w / n
    bar_w = min(28, group_w / max(2.4, s_count + 0.8))

    def y_scale(v: float) -> float:
        return margin_top + plot_h * (1.0 - (v / vmax))

    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;}</style>")
    lines.append(f"<text x='{width/2:.1f}' y='24' text-anchor='middle' font-size='16' font-weight='bold'>{escape(title)}</text>")

    # Axes
    x0, y0 = margin_left, margin_top + plot_h
    x1, y1 = margin_left + plot_w, margin_top
    lines.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='#333'/>")
    lines.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#333'/>")

    # Y ticks
    ticks = 5
    for i in range(ticks + 1):
        v = vmax * i / ticks
        y = y_scale(v)
        lines.append(f"<line x1='{x0-5}' y1='{y:.2f}' x2='{x0}' y2='{y:.2f}' stroke='#333'/>")
        lines.append(f"<text x='{x0-8}' y='{y+4:.2f}' text-anchor='end'>{v:.2f}</text>")
        lines.append(f"<line x1='{x0}' y1='{y:.2f}' x2='{x1}' y2='{y:.2f}' stroke='#eee'/>")

    # Bars
    for ci, cat in enumerate(categories):
        gx = margin_left + group_w * (ci + 0.5)
        total_w = bar_w * s_count
        start_x = gx - total_w / 2
        for si, (_sname, values) in enumerate(series):
            v = values[ci] if ci < len(values) else 0.0
            y = y_scale(v)
            h = max(0.0, y0 - y)
            x = start_x + si * bar_w
            color = colors[si % len(colors)]
            lines.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w-2:.2f}' height='{h:.2f}' fill='{color}'/>"
            )
        # x labels
        lines.append(
            f"<text x='{gx:.2f}' y='{y0+16:.2f}' text-anchor='middle' transform='rotate(20 {gx:.2f},{y0+16:.2f})'>{escape(cat)}</text>"
        )

    # Legend
    lx = margin_left
    ly = height - 26
    for si, (name, _values) in enumerate(series):
        color = colors[si % len(colors)]
        x = lx + si * 200
        lines.append(f"<rect x='{x}' y='{ly-10}' width='12' height='12' fill='{color}'/>")
        lines.append(f"<text x='{x+18}' y='{ly}'>{escape(name)}</text>")

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _fallback_svg_plots(
    out_dir: Path,
    overall_rows: List[Dict[str, Any]],
    by_type_rows: List[Dict[str, Any]],
    hybrid_embed_rows: List[Dict[str, Any]],
    type_order: List[str],
    method_order: List[str],
) -> List[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    methods = [str(r["method"]) for r in overall_rows]
    evidence = [_to_float(r["evidence_proxy_rate"]) for r in overall_rows]
    doc_div = [_to_float(r["doc_diversity_mean"]) for r in overall_rows]
    dom_div = [_to_float(r["domain_diversity_mean"]) for r in overall_rows]

    generated.append(
        _fallback_svg_bar(
            fig_dir / "overall_evidence_proxy_rate.svg",
            "Retrieval Evidence Proxy Rate by Method",
            methods,
            [("Evidence Proxy Rate", evidence)],
            y_max=1.0,
        )
    )
    generated.append(
        _fallback_svg_bar(
            fig_dir / "overall_diversity.svg",
            "Retrieval Diversity by Method",
            methods,
            [("Doc Diversity", doc_div), ("Domain Diversity", dom_div)],
        )
    )

    # Heatmaps (evidence/doc/domain) as SVG fallback.
    generated.append(
        _fallback_svg_heatmap(
            path=fig_dir / "evidence_proxy_heatmap.svg",
            title="Evidence Proxy Rate by Question Type and Method",
            row_labels=method_order,
            col_labels=type_order,
            data=_matrix_from_rows(by_type_rows, "evidence_proxy_rate", method_order, type_order),
            value_fmt="{:.2f}",
        )
    )
    generated.append(
        _fallback_svg_heatmap(
            path=fig_dir / "doc_diversity_heatmap.svg",
            title="Doc Diversity Mean by Question Type and Method",
            row_labels=method_order,
            col_labels=type_order,
            data=_matrix_from_rows(by_type_rows, "doc_diversity_mean", method_order, type_order),
            value_fmt="{:.2f}",
        )
    )
    generated.append(
        _fallback_svg_heatmap(
            path=fig_dir / "domain_diversity_heatmap.svg",
            title="Domain Diversity Mean by Question Type and Method",
            row_labels=method_order,
            col_labels=type_order,
            data=_matrix_from_rows(by_type_rows, "domain_diversity_mean", method_order, type_order),
            value_fmt="{:.2f}",
        )
    )

    hybrid_only = [r for r in hybrid_embed_rows if not str(r.get("method", "")).startswith("Delta")]
    if len(hybrid_only) >= 2:
        labels = [str(r["method"]) for r in hybrid_only]
        generated.append(
            _fallback_svg_bar(
                fig_dir / "hybrid_embedder_ablation.svg",
                "Hybrid Embedder Ablation (GTE-Qwen vs MiniLM)",
                ["Doc Diversity", "Domain Diversity", "Evidence Proxy"],
                [
                    (
                        labels[0],
                        [
                            _to_float(hybrid_only[0]["doc_diversity_mean"]),
                            _to_float(hybrid_only[0]["domain_diversity_mean"]),
                            _to_float(hybrid_only[0]["evidence_proxy_rate"]),
                        ],
                    ),
                    (
                        labels[1],
                        [
                            _to_float(hybrid_only[1]["doc_diversity_mean"]),
                            _to_float(hybrid_only[1]["domain_diversity_mean"]),
                            _to_float(hybrid_only[1]["evidence_proxy_rate"]),
                        ],
                    ),
                ],
                y_max=max(
                    1.0,
                    max(
                        _to_float(hybrid_only[0]["doc_diversity_mean"]),
                        _to_float(hybrid_only[1]["doc_diversity_mean"]),
                        _to_float(hybrid_only[0]["domain_diversity_mean"]),
                        _to_float(hybrid_only[1]["domain_diversity_mean"]),
                    ),
                ),
            )
        )

    return generated


def _matrix_from_rows(
    rows: List[Dict[str, Any]],
    metric: str,
    method_order: List[str],
    type_order: List[str],
) -> List[List[float]]:
    pivot: Dict[Tuple[str, str], float] = {}
    for r in rows:
        pivot[(str(r.get("method", "")), str(r.get("question_type", "")))] = _to_float(r.get(metric))
    return [[pivot.get((m, t), 0.0) for t in type_order] for m in method_order]


def _hex_blues(x: float) -> str:
    # Map 0..1 to a light->dark blue ramp.
    v = max(0.0, min(1.0, x))
    r = int(240 - 130 * v)
    g = int(247 - 170 * v)
    b = int(255 - 120 * v)
    return f"#{r:02x}{g:02x}{b:02x}"


def _fallback_svg_heatmap(
    *,
    path: Path,
    title: str,
    row_labels: List[str],
    col_labels: List[str],
    data: List[List[float]],
    value_fmt: str = "{:.2f}",
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(row_labels)
    n_cols = len(col_labels)
    if n_rows == 0 or n_cols == 0:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='640' height='120'></svg>", encoding="utf-8")
        return str(path)

    cell_w = 120
    cell_h = 34
    margin_left = 230
    margin_top = 90
    margin_right = 30
    margin_bottom = 40
    plot_w = n_cols * cell_w
    plot_h = n_rows * cell_h
    width = margin_left + plot_w + margin_right
    height = margin_top + plot_h + margin_bottom

    vmax = max([max(row) for row in data], default=1.0)
    if vmax <= 0:
        vmax = 1.0

    lines: List[str] = []
    lines.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    lines.append("<style>text{font-family:Arial,Helvetica,sans-serif;font-size:12px;}</style>")
    lines.append(
        f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-size='16' font-weight='bold'>{escape(title)}</text>"
    )

    # Column labels
    for j, c in enumerate(col_labels):
        x = margin_left + j * cell_w + cell_w / 2
        y = margin_top - 12
        lines.append(
            f"<text x='{x:.1f}' y='{y:.1f}' text-anchor='middle' transform='rotate(20 {x:.1f},{y:.1f})'>{escape(c)}</text>"
        )

    # Row labels + cells
    for i, rlab in enumerate(row_labels):
        y = margin_top + i * cell_h + cell_h / 2 + 4
        lines.append(f"<text x='{margin_left-10}' y='{y:.1f}' text-anchor='end'>{escape(rlab)}</text>")
        for j in range(n_cols):
            v = data[i][j] if i < len(data) and j < len(data[i]) else 0.0
            frac = v / vmax if vmax > 0 else 0.0
            color = _hex_blues(frac)
            x0 = margin_left + j * cell_w
            y0 = margin_top + i * cell_h
            lines.append(
                f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{cell_w:.1f}' height='{cell_h:.1f}' fill='{color}' stroke='#ffffff'/>"
            )
            lines.append(
                f"<text x='{x0 + cell_w/2:.1f}' y='{y0 + cell_h/2 + 4:.1f}' text-anchor='middle'>{value_fmt.format(v)}</text>"
            )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize analysis artifacts into report-ready tables/plots")
    ap.add_argument("--exp_ab_dir", default="analysis/exp_ab")
    ap.add_argument("--exp_ab_exp8_dir", default="analysis/exp_ab_exp8_minilm")
    ap.add_argument("--exp_d_dir", default="analysis/exp_d")
    ap.add_argument("--queries", default="leaderboard_queries.json")
    ap.add_argument("--out_dir", default="analysis/summary_report")
    args = ap.parse_args()

    exp_ab_dir = Path(args.exp_ab_dir)
    exp8_dir = Path(args.exp_ab_exp8_dir)
    exp_d_dir = Path(args.exp_d_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load main AB tables.
    type_counts = _read_csv(exp_ab_dir / "query_type_counts.csv")
    overall_main = _read_csv(exp_ab_dir / "summary_by_method.csv")
    by_type_main = _read_csv(exp_ab_dir / "summary_by_method_type.csv")
    details_main = _read_csv(exp_ab_dir / "retrieval_details.csv")  # optional future artifact; unused now
    _ = details_main

    # Load exp8 hybrid table (optional).
    overall_exp8 = _read_csv(exp8_dir / "summary_by_method.csv")
    by_type_exp8 = _read_csv(exp8_dir / "summary_by_method_type.csv")

    # Build overall table.
    overall_rows: List[Dict[str, Any]] = []
    for r in overall_main:
        overall_rows.append(
            {
                "method": _method_label(r.get("method", ""), source="main"),
                "n_queries": int(float(r.get("n_queries", 0) or 0)),
                "doc_diversity_mean": _to_float(r.get("doc_diversity_mean")),
                "domain_diversity_mean": _to_float(r.get("domain_diversity_mean")),
                "evidence_proxy_rate": _to_float(r.get("evidence_proxy_rate")),
            }
        )
    # Append exp8 hybrid row as separate method label.
    for r in overall_exp8:
        if str(r.get("method", "")).strip().lower() == "hybrid_rrf":
            overall_rows.append(
                {
                    "method": _method_label("hybrid_rrf", source="exp8"),
                    "n_queries": int(float(r.get("n_queries", 0) or 0)),
                    "doc_diversity_mean": _to_float(r.get("doc_diversity_mean")),
                    "domain_diversity_mean": _to_float(r.get("domain_diversity_mean")),
                    "evidence_proxy_rate": _to_float(r.get("evidence_proxy_rate")),
                }
            )

    # Build by-type table.
    by_type_rows: List[Dict[str, Any]] = []
    for r in by_type_main:
        by_type_rows.append(
            {
                "method": _method_label(r.get("method", ""), source="main"),
                "question_type": str(r.get("question_type", "")),
                "n_queries": int(float(r.get("n_queries", 0) or 0)),
                "doc_diversity_mean": _to_float(r.get("doc_diversity_mean")),
                "domain_diversity_mean": _to_float(r.get("domain_diversity_mean")),
                "evidence_proxy_rate": _to_float(r.get("evidence_proxy_rate")),
            }
        )
    for r in by_type_exp8:
        if str(r.get("method", "")).strip().lower() == "hybrid_rrf":
            by_type_rows.append(
                {
                    "method": _method_label("hybrid_rrf", source="exp8"),
                    "question_type": str(r.get("question_type", "")),
                    "n_queries": int(float(r.get("n_queries", 0) or 0)),
                    "doc_diversity_mean": _to_float(r.get("doc_diversity_mean")),
                    "domain_diversity_mean": _to_float(r.get("domain_diversity_mean")),
                    "evidence_proxy_rate": _to_float(r.get("evidence_proxy_rate")),
                }
            )

    # Hybrid embedder ablation summary.
    hybrid_qwen = next((r for r in overall_rows if r["method"] == "Hybrid (RRF, GTE-Qwen)"), None)
    hybrid_minilm = next((r for r in overall_rows if r["method"] == "Hybrid (RRF, MiniLM)"), None)
    hybrid_embed_rows: List[Dict[str, Any]] = []
    if hybrid_qwen:
        hybrid_embed_rows.append(hybrid_qwen)
    if hybrid_minilm:
        hybrid_embed_rows.append(hybrid_minilm)
    if hybrid_qwen and hybrid_minilm:
        hybrid_embed_rows.append(
            {
                "method": "Delta (GTE-Qwen - MiniLM)",
                "n_queries": hybrid_qwen["n_queries"] - hybrid_minilm["n_queries"],
                "doc_diversity_mean": hybrid_qwen["doc_diversity_mean"] - hybrid_minilm["doc_diversity_mean"],
                "domain_diversity_mean": hybrid_qwen["domain_diversity_mean"] - hybrid_minilm["domain_diversity_mean"],
                "evidence_proxy_rate": hybrid_qwen["evidence_proxy_rate"] - hybrid_minilm["evidence_proxy_rate"],
            }
        )

    # Example selection table from exp_d.
    type_map_main = {}
    # Build qid->type map from main retrieval details JSONL (more reliable than parsing queries again).
    details_jsonl_path = exp_ab_dir / "retrieval_details.jsonl"
    if details_jsonl_path.exists():
        with details_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = str(obj.get("qid", "")).strip()
                qtype = str(obj.get("question_type", "")).strip()
                if qid and qtype and qid not in type_map_main:
                    type_map_main[qid] = qtype

    example_rows = _build_example_rows(
        selected_examples_path=exp_d_dir / "selected_examples.json",
        queries_path=Path(args.queries),
        type_map_main=type_map_main,
    )

    # Write tables: CSV + Markdown.
    _write_csv(out_dir / "table_query_type_counts.csv", type_counts, ["question_type", "count"])
    _write_markdown_table(out_dir / "table_query_type_counts.md", type_counts, ["question_type", "count"])

    overall_cols = ["method", "n_queries", "doc_diversity_mean", "domain_diversity_mean", "evidence_proxy_rate"]
    _write_csv(out_dir / "table_retrieval_overall.csv", overall_rows, overall_cols)
    _write_markdown_table(out_dir / "table_retrieval_overall.md", overall_rows, overall_cols)

    by_type_cols = ["method", "question_type", "n_queries", "doc_diversity_mean", "domain_diversity_mean", "evidence_proxy_rate"]
    _write_csv(out_dir / "table_retrieval_by_type.csv", by_type_rows, by_type_cols)
    _write_markdown_table(out_dir / "table_retrieval_by_type.md", by_type_rows, by_type_cols)

    if hybrid_embed_rows:
        _write_csv(out_dir / "table_hybrid_embedder_ablation.csv", hybrid_embed_rows, overall_cols)
        _write_markdown_table(out_dir / "table_hybrid_embedder_ablation.md", hybrid_embed_rows, overall_cols)

    if example_rows:
        ex_cols = ["bucket", "qid", "question_type", "question"]
        _write_csv(out_dir / "table_expD_examples.csv", example_rows, ex_cols)
        _write_markdown_table(out_dir / "table_expD_examples.md", example_rows, ex_cols)

    # Visualization (optional, matplotlib).
    type_order = [str(r.get("question_type", "")) for r in type_counts]
    method_order = [str(r.get("method", "")) for r in overall_rows]
    figs = _try_plot_overall(
        out_dir=out_dir,
        overall_rows=overall_rows,
        by_type_rows=by_type_rows,
        hybrid_embed_rows=[r for r in hybrid_embed_rows if not str(r["method"]).startswith("Delta")],
        type_order=type_order,
        method_order=method_order,
    )

    # Master summary markdown.
    summary_lines: List[str] = []
    summary_lines.append("# Analysis Summary Bundle")
    summary_lines.append("")
    summary_lines.append("## Generated Tables")
    summary_lines.append("- `table_query_type_counts.md` / `.csv`")
    summary_lines.append("- `table_retrieval_overall.md` / `.csv`")
    summary_lines.append("- `table_retrieval_by_type.md` / `.csv`")
    if hybrid_embed_rows:
        summary_lines.append("- `table_hybrid_embedder_ablation.md` / `.csv`")
    if example_rows:
        summary_lines.append("- `table_expD_examples.md` / `.csv`")
    summary_lines.append("")
    if figs:
        summary_lines.append("## Generated Figures")
        for p in figs:
            summary_lines.append(f"- `{Path(p).name}`")
        summary_lines.append("")
    else:
        summary_lines.append("## Figures")
        summary_lines.append("- No figures generated (likely `matplotlib` unavailable).")
        summary_lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[analysis_summary] wrote summary bundle to {out_dir}")
    if figs:
        print(f"[analysis_summary] generated {len(figs)} figure(s)")
    else:
        print("[analysis_summary] no figures generated")


if __name__ == "__main__":
    main()
