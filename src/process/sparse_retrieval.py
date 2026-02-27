from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import bm25s

from utils import load_chunk_text_map, iter_jsonl, tokenize

def _bigrams(tokens: List[str], max_bigrams: int = 128) -> List[str]:
    if len(tokens) < 2:
        return []
    out = []
    for i in range(len(tokens) - 1):
        out.append(f"bg:{tokens[i]}_{tokens[i+1]}")
        if len(out) >= max_bigrams:
            break
    return out

def _minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

def build_sparse_tokens(
    chunk: dict,
    *,
    title_weight: int = 3,
    heading_weight: int = 2,
    body_weight: int = 1,
    add_bigrams: bool = True,
) -> List[str]:
    """
    Build BM25 tokens with simple field weighting:
    title > section heading > body text.
    """
    title = str(chunk.get("title", "") or "")
    heading = str(chunk.get("section_heading", "") or "")
    text = str(chunk.get("text", "") or "")

    title_tokens = tokenize(title)
    heading_tokens = tokenize(heading)
    text_tokens = tokenize(text)

    out = (
        (title_tokens * max(1, title_weight))
        + (heading_tokens * max(1, heading_weight))
        + (text_tokens * max(1, body_weight))
    )
    if add_bigrams:
        out.extend(_bigrams(title_tokens, max_bigrams=48))
        out.extend(_bigrams(heading_tokens, max_bigrams=48))
        out.extend(_bigrams(text_tokens, max_bigrams=128))
    return out

@dataclass
class SparseRetriever:
    index_dir: Path
    chunks_path: Path = Path("data/processed/chunks.jsonl")
    title_weight: int = 3
    heading_weight: int = 2
    body_weight: int = 1
    add_bigrams: bool = True

    # Pseudo-relevance feedback (PRF) query expansion
    enable_prf: bool = True
    prf_k: int = 8
    prf_terms: int = 6
    prf_alpha: float = 0.65

    def __post_init__(self):
        self.index_dir = Path(self.index_dir)
        self._bm25 = None
        self._chunk_ids: List[str] = []
        self._chunk_by_id: Dict[str, dict] | None = None

    def build(self) -> None:
        chunks = list(iter_jsonl(self.chunks_path))
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        corpus_tokens = [
            build_sparse_tokens(
                c,
                title_weight=self.title_weight,
                heading_weight=self.heading_weight,
                body_weight=self.body_weight,
                add_bigrams=self.add_bigrams,
            )
            for c in chunks
        ]

        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(self.index_dir / "bm25"))
        (self.index_dir / "chunk_ids.json").write_text(
            json.dumps(self._chunk_ids, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self) -> None:
        self._bm25 = bm25s.BM25.load(str(self.index_dir / "bm25"))
        self._chunk_ids = json.loads((self.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))

    def _ensure_chunk_cache(self) -> None:
        if self._chunk_by_id is not None:
            return
        self._chunk_by_id = {}
        for c in iter_jsonl(self.chunks_path):
            cid = str(c.get("chunk_id", ""))
            if cid:
                self._chunk_by_id[cid] = c

    def _retrieve_once(self, q_tokens: List[str], k: int) -> List[Tuple[str, float]]:
        idxs, scores = self._bm25.retrieve([q_tokens], k=k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()
        out: List[Tuple[str, float]] = []
        for idx, s in zip(idxs, scores):
            if idx is None or int(idx) < 0:
                continue
            out.append((self._chunk_ids[int(idx)], float(s)))
        return out

    def _expand_query_tokens(self, q_tokens: List[str], seed_ids: List[str]) -> List[str]:
        if not seed_ids or self.prf_terms <= 0:
            return q_tokens
        self._ensure_chunk_cache()
        assert self._chunk_by_id is not None

        qset = set(q_tokens)
        term_scores: Dict[str, float] = defaultdict(float)
        for rank, cid in enumerate(seed_ids):
            c = self._chunk_by_id.get(cid)
            if not c:
                continue
            w = 1.0 / (rank + 1.0)
            toks = build_sparse_tokens(
                c,
                title_weight=1,
                heading_weight=1,
                body_weight=1,
                add_bigrams=False,
            )
            for t in toks:
                if len(t) < 2 or t in qset:
                    continue
                term_scores[t] += w

        if not term_scores:
            return q_tokens

        expanded = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[: self.prf_terms]
        return q_tokens + [t for t, _ in expanded]

    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            raise RuntimeError("SparseRetriever not built/loaded.")

        q_tokens = tokenize(query)
        if not isinstance(q_tokens, list) or (len(q_tokens) > 0 and not isinstance(q_tokens[0], str)):
            raise ValueError("tokenize(query) must return List[str].")

        base_k = max(k, self.prf_k)
        base_res = self._retrieve_once(q_tokens, k=base_k)
        if not self.enable_prf or self.prf_k <= 0 or self.prf_terms <= 0:
            return base_res[:k]

        seed_ids = [cid for cid, _ in base_res[: self.prf_k]]
        q2_tokens = self._expand_query_tokens(q_tokens, seed_ids)
        if q2_tokens == q_tokens:
            return base_res[:k]

        prf_res = self._retrieve_once(q2_tokens, k=base_k)

        base_scores = {cid: sc for cid, sc in base_res}
        prf_scores = {cid: sc for cid, sc in prf_res}
        base_n = _minmax_norm(base_scores)
        prf_n = _minmax_norm(prf_scores)
        alpha = min(max(float(self.prf_alpha), 0.0), 1.0)

        fused: List[Tuple[str, float]] = []
        for cid in (set(base_scores) | set(prf_scores)):
            s = alpha * base_n.get(cid, 0.0) + (1.0 - alpha) * prf_n.get(cid, 0.0)
            fused.append((cid, float(s)))

        fused.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return fused[:k]

if __name__ == "__main__":
    def _minmax(v):
        if not v:
            return []
        mn, mx = min(v), max(v)
        if mx - mn < 1e-12:
            return [0.0 for _ in v]
        return [(x - mn) / (mx - mn) for x in v]

    def _retrieval_confidence(retrieved: List[Tuple[str, float]]) -> float:
        if not retrieved:
            return 0.0
        scores = [float(sc) for _cid, sc in retrieved[: min(12, len(retrieved))]]
        if len(scores) == 1:
            return 0.0
        sn = _minmax(scores)
        top = sn[0]
        second = sn[1] if len(sn) > 1 else 0.0
        gap = max(0.0, top - second)
        tail = sum(sn[2:6]) / max(1, len(sn[2:6])) if len(sn) > 2 else second
        conf = 0.55 * gap + 0.25 * top + 0.20 * max(0.0, top - tail)
        return float(min(max(conf, 0.0), 1.0))

    # Match query_ppl sparse defaults.
    k_retrieve = 80
    low_conf_k_retrieve = 160
    retrieval_conf_threshold = 0.18

    bm25_dir = Path("indexes/bm25_37763")
    if not (bm25_dir / "bm25").exists():
        retriever = SparseRetriever(
            index_dir=bm25_dir,
            title_weight=3,
            heading_weight=2,
            body_weight=1,
            add_bigrams=True,
            enable_prf=True,
            prf_k=8,
            prf_terms=6,
            prf_alpha=0.65,
        )
        retriever.build()
        print("BM25 index built at indexes/bm25")

    chunks_path = Path("data/processed/chunks_80M.jsonl")
    chunk_map = load_chunk_text_map(chunks_path)
    r = SparseRetriever(
        index_dir=bm25_dir,
        title_weight=3,
        heading_weight=2,
        body_weight=1,
        add_bigrams=True,
        enable_prf=True,
        prf_k=8,
        prf_terms=6,
        prf_alpha=0.65,
    )
    r.load()

    test_queries = [
        "Which Pittsburgh restaurant is famous for its cheesesteaks?",
        "What are the official colors of Carnegie Mellon University?",
    ]
    for q in test_queries:
        retrieved = r.retrieve(q, k=k_retrieve)
        conf = _retrieval_confidence(retrieved)
        if conf < retrieval_conf_threshold:
            retrieved = r.retrieve(q, k=low_conf_k_retrieve)

        print(f"\nQUERY: {q}")
        print(f"confidence={conf:.3f}  retrieved={len(retrieved)}")
        for cid, sc in retrieved[:5]:
            text = (chunk_map.get(cid, {}).get("text") or "").replace("\n", " ")
            print(f"  {sc:.4f}  {cid}  |  {text[:140]}")
