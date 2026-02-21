from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from hybrid_retrieval import build_default_hybrid
from utils import load_chunk_text_map

def _clean_answer(s: str) -> str:
    s = (s or "").strip()
    # remove prefixes
    for p in ["Answer:", "answer:", "A:", "Final:", "final:"]:
        if s.startswith(p):
            s = s[len(p):].strip()
    # keep first non-empty line
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        s = lines[0]
    # strip surrounding quotes
    s = s.strip().strip('"').strip("'").strip()
    return s


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
    device_map: str = "auto"
    load_in_4bit: bool = True
    max_context_tokens: int = 6000   # context budge
    max_new_tokens: int = 64
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05


class QwenReader:
    def __init__(self, cfg: ReaderConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        quant = None
        if self.cfg.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            device_map=self.cfg.device_map,
            quantization_config=quant,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()

    def build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        ctx_blocks = "\n\n".join(_format_context_chunk(i + 1, c) for i, c in enumerate(contexts))

        system = (
            "You are a question answering system.\n"
            "Answer the question based on both your knowledge and the provided context.\n"
            "Output ONLY the answer (no explanations).\n"
            "Keep your answer concise.\n"
            "Answer in English ONLY.\n"
            "No need to explain whether the answer is in context or not."
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
        dense_dir=Path("indexes/dense"),
        chunks_path=chunks_path,
        w_dense=0.6,
        w_sparse=0.4,
        k_dense=100,
        k_sparse=100,
        device="cuda",
    )

    # Reader
    reader = QwenReader(ReaderConfig(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        load_in_4bit=True,
        max_context_tokens=6000,
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9,
    ))
    reader.load()

    queries = [
        "When was Carnegie Mellon founded?",
        "Where is Picklesburgh held?",
        "What is the 2025 operating budget for Pittsburgh?",
        "Where does the Pittsburgh Symphony perform?",
    ]

    for q in queries:
        retrieved = retriever.retrieve(q, k=10)
        ctx = [chunk_map[cid] for cid, _ in retrieved if cid in chunk_map][:8]  # k_ctx=8

        ans, used = reader.answer(q, ctx)
        print("\nQ:", q)
        print("A:", ans)
        print("used:", used[:3], ("..." if len(used) > 3 else ""))
