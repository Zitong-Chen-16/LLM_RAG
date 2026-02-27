from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Union

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


@dataclass
class DenseOnlyRetriever:
    dense: DenseRetriever
    k_dense: int = 100

    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        # Keep behavior parallel to hybrid depth controls: fetch a configurable
        # dense pool, then return top-k from that pool.
        dense_k = max(int(k), int(self.k_dense))
        return self.dense.retrieve(query, k=dense_k)[:k]


@dataclass
class SparseOnlyRetriever:
    sparse: SparseRetriever
    k_sparse: int = 100
    dense: None = None

    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        # Keep behavior parallel to hybrid depth controls: fetch a configurable
        # sparse pool, then return top-k from that pool.
        sparse_k = max(int(k), int(self.k_sparse))
        return self.sparse.retrieve(query, k=sparse_k)[:k]


def build_default_dense(
    *,
    dense_dir: Path = Path("indexes/dense"),
    chunks_path: Path = Path("data/processed/chunks.jsonl"),
    k_dense: int = 100,
    device: str = "cuda",
    model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    quant_backend: str = "none",
) -> DenseOnlyRetriever:
    dense = DenseRetriever(
        index_dir=dense_dir,
        chunks_path=chunks_path,
        device=device,
        model_name=model_name,
        quant_backend=quant_backend,
    )
    dense.load()
    return DenseOnlyRetriever(dense=dense, k_dense=k_dense)


def build_default_sparse(
    *,
    bm25_dir: Path = Path("indexes/bm25"),
    chunks_path: Path = Path("data/processed/chunks.jsonl"),
    k_sparse: int = 100,
    sparse_title_weight: int = 3,
    sparse_heading_weight: int = 2,
    sparse_body_weight: int = 1,
    sparse_add_bigrams: bool = True,
    sparse_prf: bool = True,
    sparse_prf_k: int = 8,
    sparse_prf_terms: int = 6,
    sparse_prf_alpha: float = 0.65,
) -> SparseOnlyRetriever:
    sparse = SparseRetriever(
        index_dir=bm25_dir,
        chunks_path=chunks_path,
        title_weight=sparse_title_weight,
        heading_weight=sparse_heading_weight,
        body_weight=sparse_body_weight,
        add_bigrams=sparse_add_bigrams,
        enable_prf=sparse_prf,
        prf_k=sparse_prf_k,
        prf_terms=sparse_prf_terms,
        prf_alpha=sparse_prf_alpha,
    )
    sparse.load()
    return SparseOnlyRetriever(sparse=sparse, k_sparse=k_sparse)


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
    quant_backend: str = "none",
    sparse_title_weight: int = 3,
    sparse_heading_weight: int = 2,
    sparse_body_weight: int = 1,
    sparse_add_bigrams: bool = True,
    sparse_prf: bool = True,
    sparse_prf_k: int = 8,
    sparse_prf_terms: int = 6,
    sparse_prf_alpha: float = 0.65,
    fusion_method: str = "rrf",
    rrf_k: int = 60,
) -> HybridRetriever:
    sparse = SparseRetriever(
        index_dir=bm25_dir,
        chunks_path=chunks_path,
        title_weight=sparse_title_weight,
        heading_weight=sparse_heading_weight,
        body_weight=sparse_body_weight,
        add_bigrams=sparse_add_bigrams,
        enable_prf=sparse_prf,
        prf_k=sparse_prf_k,
        prf_terms=sparse_prf_terms,
        prf_alpha=sparse_prf_alpha,
    )
    sparse.load()

    dense = DenseRetriever(index_dir=dense_dir, 
                           chunks_path=chunks_path, 
                           device=device,
                           model_name=model_name,
                           quant_backend=quant_backend)
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


def build_retriever(
    *,
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    bm25_dir: Path = Path("indexes/bm25"),
    dense_dir: Path = Path("indexes/dense"),
    chunks_path: Path = Path("data/processed/chunks.jsonl"),
    w_dense: float = 0.6,
    w_sparse: float = 0.4,
    k_dense: int = 100,
    k_sparse: int = 100,
    device: str = "cuda",
    model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    quant_backend: str = "none",
    sparse_title_weight: int = 3,
    sparse_heading_weight: int = 2,
    sparse_body_weight: int = 1,
    sparse_add_bigrams: bool = True,
    sparse_prf: bool = True,
    sparse_prf_k: int = 8,
    sparse_prf_terms: int = 6,
    sparse_prf_alpha: float = 0.65,
    fusion_method: str = "rrf",
    rrf_k: int = 60,
) -> Union[HybridRetriever, DenseOnlyRetriever, SparseOnlyRetriever]:
    mode = mode.lower().strip()
    if mode == "hybrid":
        return build_default_hybrid(
            bm25_dir=bm25_dir,
            dense_dir=dense_dir,
            chunks_path=chunks_path,
            w_dense=w_dense,
            w_sparse=w_sparse,
            k_dense=k_dense,
            k_sparse=k_sparse,
            device=device,
            model_name=model_name,
            quant_backend=quant_backend,
            sparse_title_weight=sparse_title_weight,
            sparse_heading_weight=sparse_heading_weight,
            sparse_body_weight=sparse_body_weight,
            sparse_add_bigrams=sparse_add_bigrams,
            sparse_prf=sparse_prf,
            sparse_prf_k=sparse_prf_k,
            sparse_prf_terms=sparse_prf_terms,
            sparse_prf_alpha=sparse_prf_alpha,
            fusion_method=fusion_method,
            rrf_k=rrf_k,
        )

    if mode == "dense":
        return build_default_dense(
            dense_dir=dense_dir,
            chunks_path=chunks_path,
            k_dense=k_dense,
            device=device,
            model_name=model_name,
            quant_backend=quant_backend,
        )

    if mode == "sparse":
        return build_default_sparse(
            bm25_dir=bm25_dir,
            chunks_path=chunks_path,
            k_sparse=k_sparse,
            sparse_title_weight=sparse_title_weight,
            sparse_heading_weight=sparse_heading_weight,
            sparse_body_weight=sparse_body_weight,
            sparse_add_bigrams=sparse_add_bigrams,
            sparse_prf=sparse_prf,
            sparse_prf_k=sparse_prf_k,
            sparse_prf_terms=sparse_prf_terms,
            sparse_prf_alpha=sparse_prf_alpha,
        )

    raise ValueError(f"Unsupported retrieval mode: {mode}. Use hybrid|dense|sparse.")

if __name__ == "__main__":
    from query_ppl import mmr_select_chunk_ids, _retrieval_confidence

    chunks_path = Path("data/processed/chunks.jsonl")
    chunk_map = load_chunk_text_map(chunks_path)

    # Match query_ppl defaults.
    k_retrieve = 80
    stage1_k = 32
    k_ctx = 6
    mmr_lambda = 0.75
    temporal_boost_weight = 0.12
    retrieval_conf_threshold = 0.18
    low_conf_k_retrieve = 160
    low_conf_stage1_k = 56

    r = build_default_hybrid(
        bm25_dir=Path("indexes/bm25"),
        dense_dir=Path("indexes/dense_gte-Qwen2-1.5B-instruct_v2"),
        chunks_path=chunks_path,
        w_dense=0.4,
        w_sparse=0.6,
        k_dense=200,
        k_sparse=200,
        device="cuda:1",
        model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        quant_backend="none",
        sparse_title_weight=3,
        sparse_heading_weight=2,
        sparse_body_weight=1,
        sparse_add_bigrams=True,
        sparse_prf=True,
        sparse_prf_k=8,
        sparse_prf_terms=6,
        sparse_prf_alpha=0.65,
        fusion_method="rrf",
        rrf_k=60,
    )

    test_queries = [
        "Which Pittsburgh restaurant is famous for its cheesesteaks?",
        "What are the official colors of Carnegie Mellon University?",
    ]
    for q in test_queries:
        retrieved = r.retrieve(q, k=k_retrieve)
        conf = _retrieval_confidence(retrieved)
        stage1 = stage1_k
        if conf < retrieval_conf_threshold:
            retrieved = r.retrieve(q, k=low_conf_k_retrieve)
            stage1 = max(stage1, low_conf_stage1_k)

        selected_ids = mmr_select_chunk_ids(
            query=q,
            retrieved=retrieved,
            chunk_map=chunk_map,
            retriever=r,
            stage1_k=stage1,
            out_k=k_ctx,
            mmr_lambda=mmr_lambda,
            temporal_boost_weight=temporal_boost_weight,
        )

        print(f"\nQUERY: {q}")
        print(f"confidence={conf:.3f}  retrieved={len(retrieved)}  selected={len(selected_ids)}")
        for cid in selected_ids[:5]:
            text = (chunk_map.get(cid, {}).get("text") or "").replace("\n", " ")
            print(f"  {cid}  |  {text[:140]}")
