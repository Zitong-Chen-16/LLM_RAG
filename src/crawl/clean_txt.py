import re
import unicodedata
from typing import Tuple, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

# Common section headers that often start non-content regions
_STOP_SECTION_HEADERS = {
    "references",
    "notes",
    "external links",
    "see also",
    "further reading",
    "bibliography",
    "sources",
    "works cited",
    "citations",
    "footnotes",
}

# Lines that are usually boilerplate in scraped pages
_BOILERPLATE_LINE_PATTERNS = [
    r"^\s*jump to navigation\s*$",
    r"^\s*jump to search\s*$",
    r"^\s*toggle.*\s*$",
    r"^\s*contents\s*$",
    r"^\s*hide\s*$",
    r"^\s*show\s*$",
    r"^\s*this article.*(needs|lacks|may)\s+.*\s*$",
    r"^\s*from wikipedia(,)? the free encyclopedia\s*$",
]

_CITATION_BRACKET_RE = re.compile(r"\[(?:\d+|note\s*\d+|citation needed)\]", flags=re.IGNORECASE)
_MULTI_CITATION_RE = re.compile(r"\[(?:\d+(?:,\s*\d+)+)\]")
_SUPERSCRIPT_CITATION_RE = re.compile(r"(?<=\w)\s*\[\d+\](?=\s|$)")
_URL_RE = re.compile(r"https?://\S+")

# Collapse weird spacing but preserve paragraphs
_WS_RE = re.compile(r"[ \t]+")
_MANY_BLANKS_RE = re.compile(r"\n{3,}")


def clean_extracted_text(
    text: str,
    *,
    domain: Optional[str] = None,
    drop_after_stop_section: bool = True,
    remove_citations: bool = True,
    remove_urls: bool = False,
) -> str:
    """
    Clean extracted plain text from HTML/PDF.
    - Normalizes unicode
    - Fixes whitespace/newlines
    - Removes citation markers like [1], [citation needed]
    - Optionally removes trailing sections like References/External links
    - Removes common boilerplate lines
    - Lightly de-junks list/table artifacts without destroying structure
    """
    if not text:
        return ""

    # 1) Unicode normalization (fix weird punctuation and compatibility chars)
    t = unicodedata.normalize("NFKC", text)

    # 2) Standardize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # 3) Remove obvious boilerplate lines
    lines = []
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            lines.append("")
            continue

        lower = s.lower()
        if any(re.match(pat, lower) for pat in _BOILERPLATE_LINE_PATTERNS):
            continue

        # Wikipedia/HTML dumps often include "Retrieved ..." lines in reference sections
        # Keep them unless we're in stop sections (handled later).
        lines.append(ln)

    t = "\n".join(lines)

    # 4) Remove citation bracket markers
    if remove_citations:
        t = _MULTI_CITATION_RE.sub("", t)
        t = _CITATION_BRACKET_RE.sub("", t)
        t = _SUPERSCRIPT_CITATION_RE.sub("", t)

    # 5) Optionally remove URLs (sometimes they dominate "External links")
    if remove_urls:
        # Preserve sentence flow by replacing URLs with a link token instead of deleting.
        t = _URL_RE.sub(" [link] ", t)

    # 6) Drop everything after "References"/"External links"/etc. (very effective for Wikipedia)
    if drop_after_stop_section:
        # Work at paragraph boundary granularity
        paras = [p for p in t.split("\n\n")]
        kept = []
        for p in paras:
            head = p.strip().split("\n", 1)[0].strip().lower()
            # treat exact header line as stop
            if head in _STOP_SECTION_HEADERS:
                break
            kept.append(p)
        t = "\n\n".join(kept)

    # 7) Clean up whitespace: collapse inner spaces, preserve paragraph breaks
    t = "\n".join(_WS_RE.sub(" ", ln).rstrip() for ln in t.split("\n"))
    t = _MANY_BLANKS_RE.sub("\n\n", t).strip()

    # 8) Domain-specific tiny tweaks (optional)
    if domain and "wikipedia.org" in domain:
        # remove leftover edit/toolbox artifacts if present
        t = re.sub(r"\b(edit|edit source|view history)\b", "", t, flags=re.IGNORECASE)
        t = _WS_RE.sub(" ", t)

    return t.strip()

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

def _link_label_from_url(url: str) -> str:
    """
    Return a short human-readable label for a URL to avoid dropping sentence content.
    """
    s = (url or "").strip()
    if not s:
        return "[link]"
    if s.lower().startswith("www."):
        s = "https://" + s
    try:
        host = (urlparse(s).netloc or "").lower()
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    return host if host else "[link]"

def sanitize_links_in_dom(soup: BeautifulSoup) -> None:
    """
    Modify soup in-place to preserve human-readable link text while dropping raw URLs.
    """
    for a in soup.find_all("a"):
        # Visible anchor text
        anchor_text = a.get_text(" ", strip=True)
        href = (a.get("href") or "").strip()

        # If the anchor text is empty, but there is an href, don't drop surrounding sentence;
        # replace with nothing (or a single space).
        if not anchor_text:
            a.replace_with(_link_label_from_url(href))
            continue

        # If the visible text is basically a URL (or looks like it), keep no URL text.
        # Replace the anchor with cleaned anchor text (URL stripped), but keep
        # a host label when nothing readable remains.
        cleaned = URL_RE.sub("", anchor_text).strip()

        # Some sites show "doi:10...." or very long link-like strings as anchor text
        # You can add more patterns if you like.
        if cleaned == "":
            a.replace_with(_link_label_from_url(href))
        else:
            a.replace_with(cleaned)

    # Also remove URLs that appear as raw text nodes (not inside <a>)
    # We do this after anchor sanitization.
    for text_node in soup.find_all(string=True):
        s = str(text_node)
        if "http" in s or "www." in s:
            new_s = URL_RE.sub(" [link] ", s)
            if new_s != s:
                text_node.replace_with(new_s)
