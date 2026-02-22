"""
Extract plain text from:
  A) crawled data (data/raw/html/<domain>/*.html, data/raw/pdf/<domain>/*.pdf + optional manifest.jsonl)
  B) baseline folder (flat folder of *.htm/*.html files, no manifest, no domain subfolders)

Outputs:
  data/processed/docs.jsonl
"""
import argparse
import json
import os
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Iterator

from bs4 import BeautifulSoup  # pip install beautifulsoup4
import pdfplumber              # pip install pdfplumber


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def safe_domain_from_filename(name: str) -> str:
    """
    Best-effort domain inference for baseline files:
    - If filename contains ' - Wikipedia', classify as en.wikipedia.org
    """
    lower = name.lower()
    if "wikipedia" in lower:
        return "en.wikipedia.org"
    return "baseline"



def load_manifest(manifest_path: Path) -> Dict[str, dict]:
    """Map normalized saved_path -> manifest record."""
    m: Dict[str, dict] = {}
    if not manifest_path.exists():
        return m
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sp = rec.get("saved_path")
            if sp:
                m[os.path.normpath(sp)] = rec
    return m



def extract_text_from_html(html_bytes: bytes) -> Tuple[str, str]:
    """Return (title, text) from HTML bytes."""
    html = html_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe", "form"]):
        tag.decompose()

    # Remove common boilerplate containers if present
    for selector in [
        "header", "footer", "nav", "aside",
        ".nav", ".navbar", ".menu", ".footer", ".header",
        ".cookie", ".cookies", ".cookie-banner", "#cookie", "#cookies",
        ".breadcrumb", ".breadcrumbs",
        ".subscribe", ".newsletter",
        ".social", ".share", ".sharing",
        ".ad", ".ads", ".advert", ".advertisement",
    ]:
        for tag in soup.select(selector):
            tag.decompose()

    # Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    else:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)

    # Prefer <main> or <article> if present
    node = soup.find("main") or soup.find("article") or soup.body or soup
    text = node.get_text("\n", strip=True)
    return title, normalize_whitespace(text)




def extract_text_from_pdf(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        n_pages = len(pdf.pages)
        if max_pages is not None:
            n_pages = min(n_pages, max_pages)
        for i in range(n_pages):
            page = pdf.pages[i]
            t = (page.extract_text() or "").strip()
            if t:
                texts.append(t)
    return normalize_whitespace("\n\n".join(texts))



def iter_crawled_raw_files(raw_dir: Path) -> Iterator[Tuple[str, Path, str]]:
    """
    Yield (content_type, file_path, domain_dir_name) for crawled html/pdf.
    Expects:
      data/raw/html/<domain>/*.html
      data/raw/pdf/<domain>/*.pdf
    """
    html_root = raw_dir / "html"
    pdf_root = raw_dir / "pdf"

    if html_root.exists():
        for domain_dir in html_root.iterdir():
            if domain_dir.is_dir():
                domain = domain_dir.name
                for p in domain_dir.glob("*.html"):
                    yield ("text/html", p, domain)

    if pdf_root.exists():
        for domain_dir in pdf_root.iterdir():
            if domain_dir.is_dir():
                domain = domain_dir.name
                for p in domain_dir.glob("*.pdf"):
                    yield ("application/pdf", p, domain)


def iter_baseline_html_files(baseline_dir: Path) -> Iterator[Path]:
    """
    Yield baseline html paths from a flat folder.
    Accepts: .html, .htm
    """
    if not baseline_dir.exists():
        return
    for ext in ("*.html", "*.htm", "*.HTML", "*.HTM"):
        for p in baseline_dir.glob(ext):
            if p.is_file():
                yield p

def write_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw", help="Crawled raw directory")
    ap.add_argument("--manifest", default="data/raw/manifest.jsonl", help="Manifest JSONL path (optional)")
    ap.add_argument("--baseline_dir", default="data/raw/baseline_data",
                    help="Baseline folder containing flat *.htm/*.html files (no manifest).")
    ap.add_argument("--out", default="data/processed/docs.jsonl", help="Output docs JSONL")
    ap.add_argument("--max_pdf_pages", type=int, default=0, help="If >0, limit PDF extraction to first N pages")
    ap.add_argument("--min_chars", type=int, default=200, help="Skip docs with extracted text shorter than this")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    manifest_path = Path(args.manifest)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None

    if args.overwrite and out_path.exists():
        out_path.unlink()
    if args.overwrite and out_path.with_suffix(".errors.jsonl").exists():
        out_path.with_suffix(".errors.jsonl").unlink()

    manifest = load_manifest(manifest_path)
    max_pdf_pages = args.max_pdf_pages if args.max_pdf_pages > 0 else None

    n_total = n_written = n_skipped_short = n_errors = 0

    for content_type, path, domain in iter_crawled_raw_files(raw_dir):
        n_total += 1
        norm_saved = os.path.normpath(str(path))
        rec = manifest.get(norm_saved, {})

        doc_id = f"{domain}:{path.stem}"
        source_url = rec.get("url", "")
        final_url = rec.get("final_url", rec.get("url", ""))
        retrieved_at = rec.get("retrieved_at", "")
        status = rec.get("status", None)

        try:
            if content_type == "text/html":
                title, text = extract_text_from_html(path.read_bytes())
            else:
                title = ""
                text = extract_text_from_pdf(path, max_pages=max_pdf_pages)

            if len(text) < args.min_chars:
                n_skipped_short += 1
                continue

            write_jsonl(out_path, {
                "doc_id": doc_id,
                "domain": domain,
                "source_url": source_url,
                "final_url": final_url,
                "title": title,
                "text": text,
                "content_type": content_type,
                "raw_path": str(path),
                "retrieved_at": retrieved_at,
                "status": status,
                "source": "crawled",
            })
            n_written += 1
        except Exception as e:
            n_errors += 1
            write_jsonl(out_path.with_suffix(".errors.jsonl"), {
                "doc_id": doc_id,
                "domain": domain,
                "raw_path": str(path),
                "content_type": content_type,
                "error": str(e),
                "when": utc_now_iso(),
                "source": "crawled",
            })

    if baseline_dir is not None and baseline_dir.exists():
        for path in iter_baseline_html_files(baseline_dir):
            n_total += 1
            doc_hash = sha1_str(path.name)
            inferred_domain = safe_domain_from_filename(path.name)
            doc_id = f"baseline:{doc_hash}"

            try:
                title, text = extract_text_from_html(path.read_bytes())
                if not title:
                    title = path.stem

                if len(text) < args.min_chars:
                    n_skipped_short += 1
                    continue

                write_jsonl(out_path, {
                    "doc_id": doc_id,
                    "domain": inferred_domain,
                    "source_url": "",          
                    "final_url": "",           
                    "title": title,
                    "text": text,
                    "content_type": "text/html",
                    "raw_path": str(path),
                    "retrieved_at": "",        
                    "status": None,
                    "source": "baseline",
                    "baseline_filename": path.name,
                })
                n_written += 1
            except Exception as e:
                n_errors += 1
                write_jsonl(out_path.with_suffix(".errors.jsonl"), {
                    "doc_id": doc_id,
                    "domain": inferred_domain,
                    "raw_path": str(path),
                    "content_type": "text/html",
                    "error": str(e),
                    "when": utc_now_iso(),
                    "source": "baseline",
                    "baseline_filename": path.name,
                })

    print(f"[extract] scanned={n_total} written={n_written} skipped_short={n_skipped_short} errors={n_errors}")
    print(f"[extract] output={out_path}")


if __name__ == "__main__":
    main()
