import json
from pathlib import Path
from typing import List

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

def tokenize(text: str) -> List[str]:
    token_re = re.compile(r"[a-z0-9]+")
    return token_re.findall((text or "").lower())