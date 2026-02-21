from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from sparse_retrieval import SparseRetriever
from dense_retrieval import DenseRetriever
from utils import load_chunk_text_map

def minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Min-max normalize values to [0, 1]. If all equal, map all to 0.0.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


@dataclass
class HybridRetriever:
    sparse: SparseRetriever
    dense: DenseRetriever
    w_dense: float = 0.6
    w_sparse: float = 0.4
    k_dense: int = 100
    k_sparse: int = 100

    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        dense_res = self.dense.retrieve(query, k=self.k_dense)   # [(chunk_id, score)]
        sparse_res = self.sparse.retrieve(query, k=self.k_sparse)

        dense_scores = {cid: sc for cid, sc in dense_res}
        sparse_scores = {cid: sc for cid, sc in sparse_res}

        dense_norm = minmax_norm(dense_scores)
        sparse_norm = minmax_norm(sparse_scores)
        cands = set(dense_scores) | set(sparse_scores)

        fused = []
        for cid in cands:
            sd = dense_norm.get(cid, 0.0)
            ss = sparse_norm.get(cid, 0.0)
            fused_score = self.w_dense * sd + self.w_sparse * ss
            fused.append((cid, float(fused_score)))

        fused.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return fused[:k]


def build_default_hybrid(
    *,
    bm25_dir: Path = Path("indexes/bm25"),
    dense_dir: Path = Path("indexes/dense"),
    chunks_path: Path = Path("data/processed/chunks.jsonl"),
    w_dense: float = 0.6,
    w_sparse: float = 0.4,
    k_dense: int = 100,
    k_sparse: int = 100,
    device: str = "cuda",
) -> HybridRetriever:
    sparse = SparseRetriever(index_dir=bm25_dir, chunks_path=chunks_path)
    sparse.load()

    dense = DenseRetriever(index_dir=dense_dir, chunks_path=chunks_path, device=device)
    dense.load()

    return HybridRetriever(
        sparse=sparse,
        dense=dense,
        w_dense=w_dense,
        w_sparse=w_sparse,
        k_dense=k_dense,
        k_sparse=k_sparse,
    )

if __name__ == "__main__":
    chunks_path = Path("data/processed/chunks.jsonl")
    chunk_map = load_chunk_text_map(chunks_path)

    r = build_default_hybrid(
        bm25_dir=Path("indexes/bm25"),
        dense_dir=Path("indexes/dense"),
        chunks_path=chunks_path,
        w_dense=0.6,
        w_sparse=0.4,
        k_dense=100,
        k_sparse=100,
        device="cuda:1",
    )

    q = "When was the Pittsburgh Soul Food Festival established?"
    res = r.retrieve(q, k=5)
    cid, sc = res[0]
    print("\nQUERY:", q)
    print(f"  {sc:.4f}  {cid}  |  {chunk_map[cid].get('title','')}")