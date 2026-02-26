from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import numpy as np

from hybrid_retrieval import build_default_hybrid
from utils import load_chunk_text_map
from dense_retrieval import build_embed_text

#### functions from query_ppl.py that are shared with reader.py, can be refactored to utils.py if needed. ####
def dedup_by_doc_id(chunks: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for c in chunks:
        doc_id = c.get("doc_id")
        if doc_id and doc_id in seen:
            continue
        if doc_id:
            seen.add(doc_id)
        out.append(c)
        if len(out) >= k:
            break
    return out


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def mmr_select_chunk_ids(
    *,
    query: str,
    retrieved: List[tuple[str, float]],
    chunk_map: Dict[str, Dict[str, Any]],
    retriever,
    stage1_k: int,
    out_k: int,
    mmr_lambda: float,
) -> List[str]:
    cands = [(cid, float(sc)) for cid, sc in retrieved if cid in chunk_map][:stage1_k]
    if not cands:
        return []

    cand_ids = [cid for cid, _ in cands]
    fused_scores = np.array([sc for _cid, sc in cands], dtype=np.float32)
    fused_n = _minmax(fused_scores)

    dense = retriever.dense
    q_emb = dense.encode_texts([query], batch_size=1)[0]
    doc_texts = [build_embed_text(chunk_map[cid]) for cid in cand_ids]
    doc_emb = dense.encode_texts(doc_texts, batch_size=min(len(doc_texts), 64))
    dense_rel = (doc_emb @ q_emb).astype(np.float32)
    dense_rel_n = _minmax(dense_rel)

    # Blend dense relevance with hybrid prior, then enforce diversity with MMR.
    rel = (0.8 * dense_rel_n + 0.2 * fused_n).astype(np.float32)

    selected: List[int] = []
    remaining = set(range(len(cand_ids)))
    while remaining and len(selected) < out_k:
        if not selected:
            best = max(remaining, key=lambda i: float(rel[i]))
            selected.append(best)
            remaining.remove(best)
            continue

        best = None
        best_score = None
        for i in remaining:
            max_sim = max(float(doc_emb[i] @ doc_emb[j]) for j in selected)
            score = float(mmr_lambda * rel[i] - (1.0 - mmr_lambda) * max_sim)
            if best is None or score > best_score:
                best = i
                best_score = score

        selected.append(best)
        remaining.remove(best)

    return [cand_ids[i] for i in selected]
########

def _clean_answer(s: str) -> str:
    s = (s or "").strip()
    prefixes = ["Answer:", "answer:", "A:", "Final:", "final:"]

    for p in prefixes:
        if s.startswith(p):
            s = s[len(p):].strip()

    lines: List[str] = []
    for raw in s.splitlines():
        ln = raw.strip()
        if not ln:
            continue
        # Normalize common list formats so multi-answer outputs are preserved.
        ln = re.sub(r"^[-*•]\s*", "", ln)
        ln = re.sub(r"^\d+[\.\)]\s*", "", ln)
        for p in prefixes:
            if ln.startswith(p):
                ln = ln[len(p):].strip()
        ln = ln.strip().strip('"').strip("'").strip()
        if ln:
            lines.append(ln)

    if not lines:
        return ""

    if len(lines) > 1 and lines[0].rstrip(":").lower() in {"answer", "answers"}:
        lines = lines[1:]

    deduped: List[str] = []
    seen = set()
    for ln in lines:
        key = ln.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ln)
    lines = deduped or lines

    s = lines[0] if len(lines) == 1 else "; ".join(lines)
    return s.strip().strip('"').strip("'").strip()


def _format_context_chunk(i: int, c: Dict[str, Any]) -> str:
    title = (c.get("title") or "").strip()
    heading = (c.get("section_heading") or "").strip()
    url = (c.get("source_url") or "").strip()
    text = (c.get("text") or "").strip()

    meta = []
    if title:
        meta.append(f"Title: {title}")
    if heading:
        meta.append(f"Section: {heading}")
    if url:
        meta.append(f"URL: {url}")

    meta_str = " | ".join(meta) if meta else ""
    if meta_str:
        return f"[{i}] {meta_str}\n{text}"
    return f"[{i}]\n{text}"


@dataclass
class ReaderConfig:
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    device_map: Union[str, Dict[str, str]] = "cuda:0"
    quant_backend: str = "auto"  # auto | bnb | gptq | none
    load_in_4bit: bool = True     # kept for backward compatibility with bnb
    max_context_tokens: int = 4000   # context budget
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05


class QwenReader:
    def __init__(self, cfg: ReaderConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None

    def _resolve_quant_backend(self) -> str:
        backend = (self.cfg.quant_backend or "auto").strip().lower()
        if backend != "auto":
            return backend
        if "gptq" in self.cfg.model_name.lower():
            return "gptq"
        if self.cfg.load_in_4bit:
            return "bnb"
        return "none"

    def load(self) -> None:
        backend = self._resolve_quant_backend()
        quant = None
        model_kwargs: Dict[str, Any] = {
            "device_map": self.cfg.device_map,
            "trust_remote_code": True,
        }

        if backend == "bnb":
            bnb_quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            )
            model_kwargs["quantization_config"] = bnb_quant
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        elif backend == "gptq":
            # GPTQ models provide their own quantization config in model files.
            model_kwargs["torch_dtype"] = "auto"
        elif backend == "none":
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        else:
            raise ValueError(f"Unsupported quant_backend='{self.cfg.quant_backend}'. Use auto|bnb|gptq|none.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        # Some GPTQ backends (e.g., marlin post-init) do in-place tensor transforms.
        # Loading under no_grad avoids autograd leaf in-place errors.
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **model_kwargs)
        self.model.requires_grad_(False)
        self.model.eval()

    def build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        ctx_blocks = "\n\n".join(_format_context_chunk(i + 1, c) for i, c in enumerate(contexts))

        system = (
            "You are a question answering system.\n"
            "Answer the question based on the provided context.\n"
            "Output ONLY the answer (no explanations).\n"
            "KEEP YOUR ANSWER CONCISE AND NO MORE THAN ONE SENTENCE.\n"
            "Do not specify if the answer comes from context or is based on your own knowledge.\n"
            "Answer in English ONLY.\n"
        )
        user = f"Context:\n{ctx_blocks}\n\nQuestion: {question}\nAnswer:"

        # Qwen instruct format via chat template
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _truncate_contexts_to_budget(self, question: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Keep top contexts until prompt token count <= max_context_tokens (approx).
        Truncates the last context text if needed.
        """
        assert self.tokenizer is not None

        kept: List[Dict[str, Any]] = []
        for c in contexts:
            candidate = kept + [c]
            prompt = self.build_prompt(question, candidate)
            n_tok = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
            if n_tok <= self.cfg.max_context_tokens:
                kept = candidate
                continue

            # if adding this chunk exceeds budget, try truncating this chunk text
            c2 = dict(c)
            text = (c2.get("text") or "").strip()
            if not text:
                break

            # binary search truncate by characters
            lo, hi = 0, len(text)
            best = None
            while lo <= hi:
                mid = (lo + hi) // 2
                c2["text"] = text[:mid]
                prompt_mid = self.build_prompt(question, kept + [c2])
                n_mid = len(self.tokenizer(prompt_mid, add_special_tokens=False).input_ids)
                if n_mid <= self.cfg.max_context_tokens:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            if best is not None and best > 200:  # keep non-trivial text
                c2["text"] = text[:best]
                kept = kept + [c2]
            break

        return kept

    @torch.inference_mode()
    def answer(self, question: str, contexts: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Reader not loaded. Call load().")

        contexts = self._truncate_contexts_to_budget(question, contexts)
        prompt = self.build_prompt(question, contexts)

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.cfg.temperature > 0
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=self.cfg.temperature if do_sample else None,
            top_p=self.cfg.top_p if do_sample else None,
            repetition_penalty=self.cfg.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        out = self.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ans = _clean_answer(out)

        used_chunk_ids = [c.get("chunk_id", "") for c in contexts if c.get("chunk_id")]
        return ans, used_chunk_ids

if __name__ == "__main__":
    
    chunks_path = Path("data/processed/chunks.jsonl")
    chunk_map = load_chunk_text_map(chunks_path)

    # Hybrid retriever
    retriever = build_default_hybrid(
        bm25_dir=Path("indexes/bm25"),
        dense_dir=Path("indexes/dense_gte-Qwen2-1.5B-instruct"),
        chunks_path=chunks_path,
        w_dense=0.6,
        w_sparse=0.4,
        k_dense=100,
        k_sparse=100,
        device="cuda:0",
        model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",   #"Alibaba-NLP/gte-Qwen2-1.5B-instruct" | "sentence-transformers/all-MiniLM-L6-v2"
        fusion_method='rrf'
    )

    # Reader
    reader = QwenReader(ReaderConfig(
        model_name= "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", #"Qwen/Qwen2.5-14B-Instruct",
        device_map={"": "cuda:1"},
        quant_backend="gptq",
        max_context_tokens=24000,
        max_new_tokens=48,
        temperature=0,
        top_p=1,
    ))
    reader.load()

    queries = [
        "What is the name of Carnegie Mellon's student newspaper?",
        "What are the official colors of Carnegie Mellon University?",
        "Which Pittsburgh restaurant is famous for its cheesesteaks?",
        "What is the specialty of Tessaro’s in Pittsburgh?",
        "Which Pittsburgh bakery is known for its macarons?",
        "What is the specialty of Fet Fisk in Pittsburgh?",
    ]

    for q in queries:
        retrieved = retriever.retrieve(q, k=50)
        selected_ids = mmr_select_chunk_ids(
            query=q,
            retrieved=retrieved,
            chunk_map=chunk_map,
            retriever=retriever,
            stage1_k=50,
            out_k=10,
            mmr_lambda=0.75,
        )
        ctx = [chunk_map[cid] for cid in selected_ids if cid in chunk_map]
        if len(ctx) < 6:
            seen_ids = set(selected_ids)
            for cid, _sc in retrieved:
                if cid in chunk_map and cid not in seen_ids:
                    ctx.append(chunk_map[cid])
                    seen_ids.add(cid)
                    if len(ctx) >= 6:
                        break
        dedup_doc = False
        if dedup_doc:
            ctx = dedup_by_doc_id(ctx, 6)
        ctx = [chunk_map[cid] for cid, _ in retrieved if cid in chunk_map]  # k_ctx=6
        ans, _used = reader.answer(q, ctx)
        ans = (ans or "").strip()

        ans, used = reader.answer(q, ctx)
        print("\nQ:", q)
        print("A:", ans)
        print("used:", used[:3], ("..." if len(used) > 3 else ""))
