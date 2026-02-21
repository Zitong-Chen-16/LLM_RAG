from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import bm25s

from utils import load_chunk_text_map, iter_jsonl, tokenize

@dataclass
class SparseRetriever:
    index_dir: Path
    chunks_path: Path = Path("data/processed/chunks.jsonl")

    def __post_init__(self):
        self.index_dir = Path(self.index_dir)
        self._bm25 = None
        self._chunk_ids: List[str] = []

    def build(self) -> None:
        chunks = list(iter_jsonl(self.chunks_path))
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        corpus_tokens = [tokenize(c["text"]) for c in chunks]

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

    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            raise RuntimeError("SparseRetriever not built/loaded.")

        q_tokens = tokenize(query)
        if not isinstance(q_tokens, list) or (len(q_tokens) > 0 and not isinstance(q_tokens[0], str)):
            raise ValueError("tokenize(query) must return List[str].")

        idxs, scores = self._bm25.retrieve([q_tokens], k=k)

        # scores/doc_ids are (1, k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()
        return [(self._chunk_ids[int(idx)], float(s)) for idx, s in zip(idxs, scores)]

if __name__ == "__main__":
    

    if not (Path("indexes/bm25") / "bm25").exists():
        retriever = SparseRetriever(index_dir=Path("indexes/bm25"))
        retriever.build()
        print("BM25 index built at indexes/bm25")

    # test retrieval
    chunks_path = Path("data/processed/chunks.jsonl")
    chunk_map = load_chunk_text_map(chunks_path)
    
    r = SparseRetriever(index_dir=Path("indexes/bm25"))
    r.load()

    query = "When was the Pittsburgh Soul Food Festival established?"
    results = r.retrieve(query, k=2)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    top_chunk_id, top_score = results[0]
    top = chunk_map[top_chunk_id]

    print(f"Query: {query}")
    print(f"Top chunk: {top_chunk_id}  score={top_score:.4f}")
    print(f"Title: {top.get('title','')}")
    print(f"Section: {top.get('section_heading')}")
    print("-" * 80)
    print(top["text"])