import json
from pathlib import Path
from typing import List
import re
import unicodedata

def load_chunk_text_map(chunks_path: Path):
    m = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            m[obj["chunk_id"]] = obj
    return m

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[’'\\-/][a-z0-9]+)*", flags=re.IGNORECASE)
SPLIT_INNER_RE = re.compile(r"[’'\\-/]+")

def tokenize(text: str) -> List[str]:
    """
    Robust lexical tokenizer for sparse retrieval.
    - Keeps alnum tokens
    - Preserves hyphen/apostrophe/slash compounds (e.g., "carnegie-mellon")
    - Adds split sub-tokens (e.g., "carnegie", "mellon") for recall
    """
    t = unicodedata.normalize("NFKC", text or "").lower()
    out: List[str] = []
    for tok in TOKEN_RE.findall(t):
        tok = tok.replace("’", "'").strip("-'/")
        if not tok:
            continue
        out.append(tok)
        if any(ch in tok for ch in ("-", "'", "/")):
            parts = [p for p in SPLIT_INNER_RE.split(tok) if p]
            if len(parts) > 1:
                out.extend(parts)
    return out
