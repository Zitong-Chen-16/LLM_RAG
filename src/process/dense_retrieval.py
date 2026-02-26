import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import faiss 
import torch
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig, AutoConfig

from utils import iter_jsonl, load_chunk_text_map

def build_embed_text(chunk: dict) -> str:
    title = (chunk.get("title") or "").strip()
    heading = (chunk.get("section_heading") or "").strip()
    text = (chunk.get("text") or "").strip()

    parts = []
    if title:
        parts.append(title)
    if heading:
        parts.append(heading)
    if text:
        parts.append(text)
    return "\n".join(parts)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


@dataclass
class DenseRetriever:
    index_dir: Path
    chunks_path: Path = Path("data/processed/chunks.jsonl")
    model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    device: str = "cuda"
    batch_size: int = 8
    normalize: bool = True
    quant_backend: str = "none"  # none | 8bit | 4bit
    query_instruction: Optional[str] = None

    def __post_init__(self):
        self.index_dir = Path(self.index_dir)
        self._model: Optional[SentenceTransformer] = None
        self._index = None
        self._chunk_ids: List[str] = []

    def _default_query_instruction(self) -> Optional[str]:
        name = (self.model_name or "").lower()
        if "gte-qwen2" in name and "instruct" in name:
            return "Given a web search query, retrieve relevant passages that answer the query"
        return None

    def _get_query_instruction(self) -> Optional[str]:
        if self.query_instruction and self.query_instruction.strip():
            return self.query_instruction.strip()
        return self._default_query_instruction()

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            backend = (self.quant_backend or "none").strip().lower()
            try:
                cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            except Exception:
                cfg = None

            def build_model(trust_remote_code: bool) -> SentenceTransformer:
                st_kwargs = {"trust_remote_code": trust_remote_code}
                if cfg is not None and getattr(cfg, "model_type", "") == "qwen2" and not hasattr(cfg, "rope_theta"):
                    st_kwargs["config_kwargs"] = {"rope_theta": 1000000.0}

                if backend == "none":
                    return SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        **st_kwargs,
                    )
                if backend in {"8bit", "4bit"}:
                    if backend == "8bit":
                        quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    else:
                        quant_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                        )
                    return SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        **st_kwargs,
                        model_kwargs={
                            "quantization_config": quant_config,
                            "device_map": {"": self.device},
                            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                        },
                    )
                raise ValueError(f"Unsupported quant_backend='{self.quant_backend}'. Use none|8bit|4bit.")

            try:
                self._model = build_model(trust_remote_code=True)
            except AttributeError as e:
                if "rope_theta" not in str(e):
                    raise
                self._model = build_model(trust_remote_code=False)

        return self._model

    def build(self, save_embeddings: bool = False) -> None:
        chunks = list(iter_jsonl(self.chunks_path))
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [build_embed_text(c) for c in chunks]

        emb = self.encode_documents(texts, batch_size=self.batch_size, show_progress_bar=True)

        if self.normalize:
            emb = l2_normalize(emb)

        d = emb.shape[1]
        index = faiss.IndexFlatIP(d) 
        index.add(emb)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_dir / "faiss.index"))
        (self.index_dir / "chunk_ids.json").write_text(
            json.dumps(self._chunk_ids, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.index_dir / "meta.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "device": self.device,
                    "batch_size": self.batch_size,
                    "normalize": self.normalize,
                    "quant_backend": self.quant_backend,
                    "dim": int(d),
                    "num_chunks": int(len(self._chunk_ids)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if save_embeddings:
            np.save(self.index_dir / "embeddings.npy", emb)

        self._index = index

    def load(self) -> None:
        self._index = faiss.read_index(str(self.index_dir / "faiss.index"))
        self._chunk_ids = json.loads((self.index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        meta_path = self.index_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            built_model_name = str(meta.get("model_name", "")).strip()
            if built_model_name and built_model_name != self.model_name:
                raise ValueError(
                    f"Dense index model mismatch. Index built with '{built_model_name}', "
                    f"but retriever configured with '{self.model_name}'. Rebuild the index or use matching model."
                )

    def _encode_raw(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        model = self._load_model()
        kwargs = dict(extra_kwargs or {})
        emb = model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=False,
            **kwargs,
        ).astype(np.float32)
        if self.normalize:
            emb = l2_normalize(emb)
        return emb

    def encode_documents(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        # Documents should be encoded without query instructions/prompts.
        return self._encode_raw(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            extra_kwargs=None,
        )

    def encode_queries(
        self,
        queries: List[str],
        *,
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        instruction = self._get_query_instruction()
        model = self._load_model()

        # Preferred path for instruction-tuned embedders with built-in prompts.
        prompts = getattr(model, "prompts", None)
        if isinstance(prompts, dict) and "query" in prompts:
            try:
                return self._encode_raw(
                    queries,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    extra_kwargs={"prompt_name": "query"},
                )
            except Exception:
                pass

        if instruction:
            # Fallback to explicit prompt prefix.
            prompt_prefix = f"Instruct: {instruction}\nQuery: "
            try:
                return self._encode_raw(
                    queries,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    extra_kwargs={"prompt": prompt_prefix},
                )
            except TypeError:
                # Older sentence-transformers versions may not support `prompt=`.
                formatted = [f"{prompt_prefix}{q}" for q in queries]
                return self._encode_raw(
                    formatted,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    extra_kwargs=None,
                )

        return self._encode_raw(
            queries,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            extra_kwargs=None,
        )

    def encode_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        # Backward-compatible alias: treat this as document encoding.
        return self.encode_documents(texts, batch_size=batch_size, show_progress_bar=False)

    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]:
        q = self.encode_queries([query], batch_size=1)

        scores, idxs = self._index.search(q, k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        out: List[Tuple[str, float]] = []
        for s, idx in zip(scores, idxs):
            if idx < 0:
                continue
            out.append((self._chunk_ids[int(idx)], float(s)))
        return out

if __name__ == "__main__":

    if not (Path("indexes/dense") / "faiss.index").exists():
        r = DenseRetriever(
            index_dir=Path("indexes/dense"),
            chunks_path=Path("data/processed/chunks.jsonl"),
            model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
            device="cuda:1",
            batch_size=8,
            normalize=True,
            quant_backend="8bit"
            )
        r.build(save_embeddings=False)
        print("Dense FAISS index built at indexes/dense")

    # test retrieval
    chunks_path = Path("data/processed/chunks.jsonl")
    chunk_map = load_chunk_text_map(chunks_path)

    r = DenseRetriever(index_dir=Path("indexes/dense"), 
                       chunks_path=chunks_path, 
                       model_name="Alibaba-NLP/gte-Qwen2-7B-instruct", #Alibaba-NLP/gte-Qwen2-1.5B-instruct
                       device="cuda:1")
    r.load()

    
    q = "Which Pittsburgh restaurant is famous for its cheesesteaks?"

    res = r.retrieve(q, k=5)

    cid, sc = res[0]
    print("\nQUERY:", q)
    print(f"  {sc:.4f}  {cid}  |  {chunk_map[cid].get('text','')}")        
