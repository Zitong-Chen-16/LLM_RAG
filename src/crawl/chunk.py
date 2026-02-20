"""
Chunking policy:
- Target chunk size: 250–400 words (soft target) or 800–1200 chars (secondary guard)
- Overlap: 50–100 words
- Boundary-aware:
  1) split by headings/paragraphs first
  2) if paragraph too long -> split by sentences
  3) if still too long -> fallback to fixed-size word windowing
- Edge cases:
  - very short docs -> single chunk
  - tables / bullet lists -> keep list blocks together when possible
"""
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

@dataclass
class ChunkConfig:
    min_words: int = 200          
    target_words: int = 320       
    min_words_chunk: int = 250
    max_words_chunk: int = 400
    overlap_words: int = 80       

    min_chars_chunk: int = 800    
    max_chars_chunk: int = 1200

    long_para_words: int = 450
    long_sentence_words: int = 120
    window_words: int = 320


def word_count(s: str) -> int:
    return len(re.findall(r"\S+", s))


def split_into_lines(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    return lines


def is_bullet_line(line: str) -> bool:
    bullet = re.compile(r"^\s*(?:[-*•]|(\d+[\.\)]))\s+")
    return bool(bullet.match(line))


def is_table_line(line: str) -> bool:
    table = re.compile(r"^\s*\|.*\|\s*$")
    return bool(table.match(line))


def is_heading_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if is_bullet_line(s) or is_table_line(s):
        return False
    heading = re.compile(r"^[A-Z0-9][A-Z0-9\s\-\–\—:]{6,}$")
    if len(s) <= 80 and not s.endswith(".") and (s.endswith(":") or heading.match(s)):
        return True
    if len(s) <= 60 and not s.endswith(".") and sum(ch.isalpha() for ch in s) >= 8:
        if s.count(" ") <= 8:
            return True
    return False


def sentence_split(paragraph: str) -> List[str]:
    """
    Lightweight sentence splitter: splits on . ! ? followed by whitespace + capital/quote/paren.
    Not perfect, but good enough for chunk boundaries.
    """
    p = paragraph.strip()
    if not p:
        return []
    protected = re.sub(r"\b(e\.g|i\.e|Mr|Mrs|Ms|Dr|St|vs)\.", lambda m: m.group(0).replace(".", "<DOT>"), p)

    parts = re.split(r"(?<=[\.\!\?])\s+(?=[\"'\(\[]?[A-Z0-9])", protected)
    out = []
    for part in parts:
        part = part.replace("<DOT>", ".").strip()
        if part:
            out.append(part)
    return out


def join_blocks(blocks: List[str]) -> str:
    return "\n".join(b.strip() for b in blocks if b.strip()).strip()


def segment_blocks(text: str) -> List[Tuple[Optional[str], str]]:
    lines = split_into_lines(text)

    blocks: List[Tuple[Optional[str], str]] = []
    curr_heading: Optional[str] = None

    buf: List[str] = []
    buf_kind: Optional[str] = None  # "para" | "list" | "table"

    def flush():
        nonlocal buf, buf_kind
        if buf:
            blocks.append((curr_heading, join_blocks(buf)))
        buf = []
        buf_kind = None

    for ln in lines:
        s = ln.strip()

        if not s:
            if buf_kind == "para":
                flush()
            elif buf_kind in ("list", "table"):
                buf.append("") 
            continue

        # detect headings
        if is_heading_line(s):
            flush()
            curr_heading = s.rstrip(":").strip()
            continue

        # detect list/table lines
        if is_bullet_line(s):
            if buf_kind not in ("list", None):
                flush()
            buf_kind = "list"
            buf.append(s)
            continue

        if is_table_line(s):
            if buf_kind not in ("table", None):
                flush()
            buf_kind = "table"
            buf.append(s)
            continue

        # normal paragraph line
        if buf_kind in ("list", "table"):
            flush()

        if buf_kind not in ("para", None):
            flush()
        buf_kind = "para"
        buf.append(s)

    flush()
    return blocks


def window_words(words: List[str], size: int, overlap: int) -> List[str]:
    """Return list of window texts with overlap."""
    out = []
    i = 0
    n = len(words)
    step = max(1, size - overlap)
    while i < n:
        j = min(n, i + size)
        out.append(" ".join(words[i:j]).strip())
        if j == n:
            break
        i += step
    return out


def split_block_if_needed(block: str, cfg: ChunkConfig) -> List[str]:
    """
    Split a block if it's too long. Prefer sentence splitting, fallback to windowing.
    """
    w = word_count(block)
    if w <= cfg.long_para_words:
        return [block]

    # sentence split
    sents = []
    for part in sentence_split(block):
        if word_count(part) > cfg.long_sentence_words:
            words = re.findall(r"\S+", block)
            return window_words(words, cfg.window_words, cfg.overlap_words)

        sents.append(part)

    #  pack sentences into smaller blocks
    packed: List[str] = []
    cur: List[str] = []
    cur_w = 0
    for sent in sents:
        sw = word_count(sent)
        if cur and cur_w + sw > cfg.max_words_chunk:
            packed.append(" ".join(cur).strip())
            cur = [sent]
            cur_w = sw
        else:
            cur.append(sent)
            cur_w += sw
    if cur:
        packed.append(" ".join(cur).strip())

    out: List[str] = []
    for p in packed:
        if word_count(p) > cfg.max_words_chunk + 50:
            words = re.findall(r"\S+", p)
            out.extend(window_words(words, cfg.window_words, cfg.overlap_words))
        else:
            out.append(p)
    return out


def pack_blocks_into_chunks(
    blocks: List[Tuple[Optional[str], str]],
    cfg: ChunkConfig
) -> List[Tuple[Optional[str], str, int, int]]:
    """
    Pack (heading, block_text) sequence into chunks.
    Returns list of (section_heading, chunk_text, char_start, char_end) relative to doc text.
    char_start/end are best-effort over concatenated blocks (not exact original raw chars).
    """
    expanded: List[Tuple[Optional[str], str]] = []
    for heading, block in blocks:
        for piece in split_block_if_needed(block, cfg):
            expanded.append((heading, piece))

    chunks: List[Tuple[Optional[str], str, int, int]] = []
    cur_parts: List[str] = []
    cur_heading: Optional[str] = None
    cur_words = 0

    pos = 0
    cur_start = 0

    def flush_chunk():
        nonlocal cur_parts, cur_words, cur_heading, cur_start, pos
        if not cur_parts:
            return
        text = "\n\n".join(cur_parts).strip()
        if text:
            char_end = cur_start + len(text)
            chunks.append((cur_heading, text, cur_start, char_end))
            pos = char_end + 2
        cur_parts = []
        cur_words = 0
        cur_heading = None
        cur_start = pos

    for heading, block in expanded:
        bw = word_count(block)

        if cur_heading is None:
            cur_heading = heading

        if bw > cfg.max_words_chunk + 50:
            flush_chunk()
            words = re.findall(r"\S+", block)
            for wtxt in window_words(words, cfg.window_words, cfg.overlap_words):
                chunks.append((heading, wtxt, pos, pos + len(wtxt)))
                pos += len(wtxt) + 2
            cur_start = pos
            continue

        if cur_parts and (cur_words + bw > cfg.max_words_chunk):
            flush_chunk()
            cur_heading = heading

        cur_parts.append(block)
        cur_words += bw

        if cur_words >= cfg.min_words_chunk and len("\n\n".join(cur_parts)) >= cfg.max_chars_chunk:
            flush_chunk()

    flush_chunk()

    # Apply overlap 
    if cfg.overlap_words > 0 and len(chunks) >= 2:
        overlapped: List[Tuple[Optional[str], str, int, int]] = []
        prev_words: List[str] = []
        for i, (heading, text, cs, ce) in enumerate(chunks):
            words = re.findall(r"\S+", text)
            if i == 0:
                overlapped.append((heading, text, cs, ce))
            else:
                overlap = prev_words[-cfg.overlap_words:] if len(prev_words) >= cfg.overlap_words else prev_words
                new_text = (" ".join(overlap).strip() + "\n\n" + text).strip() if overlap else text
                overlapped.append((heading, new_text, cs, cs + len(new_text)))
            prev_words = words
        chunks = overlapped

    return chunks


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_docs", default="data/processed/docs.jsonl")
    ap.add_argument("--out_chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--overwrite", action="store_true")
    # main knobs
    ap.add_argument("--min_words_chunk", type=int, default=250)
    ap.add_argument("--max_words_chunk", type=int, default=400)
    ap.add_argument("--overlap_words", type=int, default=80)
    ap.add_argument("--min_chars_chunk", type=int, default=800)
    ap.add_argument("--max_chars_chunk", type=int, default=1200)
    ap.add_argument("--min_doc_words_single_chunk", type=int, default=200)
    args = ap.parse_args()

    cfg = ChunkConfig(
        min_words=args.min_doc_words_single_chunk,
        min_words_chunk=args.min_words_chunk,
        max_words_chunk=args.max_words_chunk,
        overlap_words=args.overlap_words,
        min_chars_chunk=args.min_chars_chunk,
        max_chars_chunk=args.max_chars_chunk,
        target_words=(args.min_words_chunk + args.max_words_chunk) // 2,
    )

    in_path = Path(args.in_docs)
    out_path = Path(args.out_chunks)
    if args.overwrite and out_path.exists():
        out_path.unlink()

    n_docs = 0
    n_chunks = 0

    for doc in iter_jsonl(in_path):
        n_docs += 1
        doc_id = doc.get("doc_id", "")
        text = doc.get("text", "") or ""
        title = doc.get("title", "") or ""
        source_url = doc.get("source_url", "") or ""
        domain = doc.get("domain", "") or ""

        # If doc is short, keep as one chunk
        if word_count(text) < cfg.min_words:
            chunk_text = text.strip()
            if chunk_text:
                chunk_id = f"{doc_id}:c0"
                write_jsonl(out_path, {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "domain": domain,
                    "source_url": source_url,
                    "title": title,
                    "section_heading": None,
                    "text": chunk_text,
                    "char_start": 0,
                    "char_end": len(chunk_text),
                })
                n_chunks += 1
            continue

        blocks = segment_blocks(text)
        packed = pack_blocks_into_chunks(blocks, cfg)

        for i, (heading, chunk_text, cs, ce) in enumerate(packed):
            chunk_id = f"{doc_id}:c{i}"
            write_jsonl(out_path, {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "domain": domain,
                "source_url": source_url,
                "title": title,
                "section_heading": heading,
                "text": chunk_text,
                "char_start": cs,
                "char_end": ce,
            })
            n_chunks += 1

    print(f"[chunk] docs={n_docs} chunks={n_chunks} out={out_path}")


if __name__ == "__main__":
    main()
