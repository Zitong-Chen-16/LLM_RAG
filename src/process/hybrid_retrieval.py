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
    fusion_method: str = "rrf"
    rrf_k: int = 60

    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        dense_res = self.dense.retrieve(query, k=self.k_dense)   # [(chunk_id, score)]
        sparse_res = self.sparse.retrieve(query, k=self.k_sparse)

        dense_scores = {cid: sc for cid, sc in dense_res}
        sparse_scores = {cid: sc for cid, sc in sparse_res}
        cands = set(dense_scores) | set(sparse_scores)

        fused = []
        if self.fusion_method.lower() == "rrf":
            dense_rank = {cid: i + 1 for i, (cid, _sc) in enumerate(dense_res)}
            sparse_rank = {cid: i + 1 for i, (cid, _sc) in enumerate(sparse_res)}
            for cid in cands:
                rd = dense_rank.get(cid)
                rs = sparse_rank.get(cid)
                sd = (self.w_dense / (self.rrf_k + rd)) if rd is not None else 0.0
                ss = (self.w_sparse / (self.rrf_k + rs)) if rs is not None else 0.0
                fused.append((cid, float(sd + ss)))
        elif self.fusion_method.lower() == 'minmax':
            dense_norm = minmax_norm(dense_scores)
            sparse_norm = minmax_norm(sparse_scores)
            for cid in cands:
                sd = dense_norm.get(cid, 0.0)
                ss = sparse_norm.get(cid, 0.0)
                fused_score = self.w_dense * sd + self.w_sparse * ss
                fused.append((cid, float(fused_score)))
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
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
    model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    fusion_method: str = "rrf",
    rrf_k: int = 60,
) -> HybridRetriever:
    sparse = SparseRetriever(index_dir=bm25_dir, chunks_path=chunks_path)
    sparse.load()

    dense = DenseRetriever(index_dir=dense_dir, 
                           chunks_path=chunks_path, 
                           device=device,
                            model_name=model_name)
    dense.load()

    return HybridRetriever(
        sparse=sparse,
        dense=dense,
        w_dense=w_dense,
        w_sparse=w_sparse,
        k_dense=k_dense,
        k_sparse=k_sparse,
        fusion_method=fusion_method,
        rrf_k=rrf_k,
    )

if __name__ == "__main__":
    chunks_path = Path("data/processed/chunks.jsonl")
    chunk_map = load_chunk_text_map(chunks_path)

    r = build_default_hybrid(
        bm25_dir=Path("indexes/bm25"),
        dense_dir=Path("indexes/dense_gte-Qwen2-1.5B-instruct"),
        chunks_path=chunks_path,
        w_dense=0.6,
        w_sparse=0.4,
        k_dense=100,
        k_sparse=100,
        device="cuda:0",
    )

    q = "Which Pittsburgh restaurant is famous for its cheesesteaks?"
    res = r.retrieve(q, k=5)
    cid, sc = res[0]
    print("\nQUERY:", q)
    print(f"  {sc:.4f}  {cid}  |  {chunk_map[cid].get('text','')}")
