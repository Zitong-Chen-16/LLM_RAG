import hashlib, json, os, re, time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag
import os
import requests
import argparse

from bs4 import BeautifulSoup
import yaml


@dataclass
class CrawlConfig:
    domain: str
    seeds: list[str]
    out_dir: str = "data/raw"
    max_pages: int = 2000
    max_depth: int = 2
    delay_s: float = 1.0
    allow_patterns: list[str] = None
    deny_patterns: list[str] = None

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def canonicalize(url: str) -> str:
    url, _frag = urldefrag(url)
    # strip common tracking params if you want
    return url

def allowed(url: str, cfg: CrawlConfig) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if not p.netloc.endswith(cfg.domain):
        return False
    path = p.path.lower()

    if cfg.deny_patterns:
        for pat in cfg.deny_patterns:
            if re.search(pat, url, flags=re.I):
                return False

    if cfg.allow_patterns:
        return any(re.search(pat, url, flags=re.I) for pat in cfg.allow_patterns)

    # default: allow everything in-domain
    return True

def save_bytes(path: str, b: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)

def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def extract_links(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        links.append(urljoin(base_url, href))
    return links

def fetch(url: str, session: requests.Session) -> tuple[int, str, bytes]:
    resp = session.get(url, timeout=30, allow_redirects=True)
    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    return resp.status_code, ctype, resp.content

def crawl_domain(cfg: CrawlConfig):
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; CMU-ANLP-RAG-Scraper/1.0; +https://cmu.edu)"
    })

    q = [(canonicalize(u), 0) for u in cfg.seeds]
    seen = set()
    n = 0

    manifest_path = os.path.join(cfg.out_dir, "manifest.jsonl")

    while q and n < cfg.max_pages:
        url, depth = q.pop(0)
        if url in seen:
            continue
        seen.add(url)

        if depth > cfg.max_depth or not allowed(url, cfg):
            continue

        time.sleep(cfg.delay_s)

        try:
            status, ctype, content = fetch(url, session)
        except Exception as e:
            append_jsonl(manifest_path, {
                "url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            })
            continue

        rec = {
            "url": url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "content_type": ctype,
        }

        if status != 200:
            append_jsonl(manifest_path, rec)
            continue

        domain_dir = os.path.join(cfg.out_dir, "html" if ctype.startswith("text/html") else "pdf", cfg.domain)
        fname = sha1(url)

        if ctype.startswith("text/html"):
            path = os.path.join(domain_dir, f"{fname}.html")
            save_bytes(path, content)
            rec["saved_path"] = path
            # enqueue links
            try:
                html = content.decode("utf-8", errors="ignore")
                for link in extract_links(url, html):
                    link = canonicalize(link)
                    if link not in seen and allowed(link, cfg):
                        q.append((link, depth + 1))
            except Exception:
                pass

        elif ctype == "application/pdf" or url.lower().endswith(".pdf"):
            path = os.path.join(domain_dir, f"{fname}.pdf")
            save_bytes(path, content)
            rec["saved_path"] = path

        else:
            # skip other content types (images/js/etc.)
            rec["skipped"] = True

        append_jsonl(manifest_path, rec)
        n += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/crawl_config.yaml",
        help="Path to crawl YAML config"
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated list of domains to crawl (e.g., visitpittsburgh.com,events.cmu.edu). "
             "If omitted, crawls all domains in the config."
    )
    args = parser.parse_args()

    # Load YAML
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)

    out_dir = cfg_all.get("out_dir", "data/raw/additional")
    global_deny = cfg_all.get("global_deny_patterns", [])

    # Optional domain filter
    only = None
    if args.domains:
        only = {d.strip() for d in args.domains.split(",") if d.strip()}

    # Iterate domains
    for dcfg in cfg_all.get("domains", []):
        domain = dcfg["domain"]
        if only is not None and domain not in only:
            continue

        # Merge global + per-domain deny patterns
        deny_patterns = list(global_deny) + list(dcfg.get("deny_patterns", []))

        cfg = CrawlConfig(
            domain=domain,
            seeds=dcfg["seeds"],
            out_dir=out_dir,                
            max_pages=int(dcfg.get("max_pages", 500)),
            max_depth=int(dcfg.get("max_depth", 2)),
            delay_s=float(dcfg.get("delay_s", 1.0)),
            allow_patterns=dcfg.get("allow_patterns", []),
            deny_patterns=deny_patterns,
        )

        print(f"[crawl] domain={domain} seeds={len(cfg.seeds)} max_pages={cfg.max_pages} max_depth={cfg.max_depth}")
        crawl_domain(cfg)

