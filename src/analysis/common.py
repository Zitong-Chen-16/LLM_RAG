from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

MONTH_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\b",
    flags=re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b|\b(?:am|pm)\b", flags=re.IGNORECASE)
DOMAIN_RE = re.compile(r"\b[a-z0-9][a-z0-9.-]+\.(?:com|org|edu|gov|net|io)\b", flags=re.IGNORECASE)
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[a-z0-9.\-'\s]{2,40}\s(?:st|street|ave|avenue|rd|road|blvd|boulevard|"
    r"dr|drive|ln|lane|way|pkwy|parkway)\b",
    flags=re.IGNORECASE,
)
DIGIT_RE = re.compile(r"\d")
LIST_MARKER_RE = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+\.)\s+", flags=re.IGNORECASE)

EVENT_DATE_TERMS = {
    "when", "date", "month", "year", "schedule", "upcoming", "event", "events",
    "festival", "concert", "performing", "performance", "show", "calendar",
}
NUMERIC_TERMS = {
    "how many", "how much", "number of", "population", "percent", "percentage",
    "rank", "ranking", "cost", "price", "distance", "area", "size", "amount",
}
LIST_TERMS = {
    "what are the names", "name the", "list", "which are", "what are the",
}
LOCATION_TERMS = {"where", "located", "location", "address", "in pittsburgh"}
WEBSITE_TERMS = {"website", "url", "site", "webpage", "web page"}


def load_queries(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list at {path}")
    out: List[Dict[str, str]] = []
    for item in data:
        qid = str(item.get("id", "")).strip()
        q = str(item.get("question", "")).strip()
        if qid and q:
            out.append({"id": qid, "question": q})
    return out


def load_answers(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            key = str(k).strip()
            if not key.isdigit():
                continue
            out[key] = str(v or "").strip()
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def snippet(text: str, max_chars: int = 180) -> str:
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)].rstrip() + "…"


def classify_question_type(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "description"

    has_event_date = (
        bool(MONTH_RE.search(q))
        or bool(YEAR_RE.search(q))
        or any(t in q for t in EVENT_DATE_TERMS)
    )
    if has_event_date:
        return "event_date"

    if any(t in q for t in NUMERIC_TERMS):
        return "numeric"

    if any(t in q for t in LIST_TERMS):
        return "list"

    if any(t in q for t in LOCATION_TERMS) or any(t in q for t in WEBSITE_TERMS):
        return "entity_location"

    if q.startswith(("who ", "where ", "which ", "what is the name", "what is ")):
        return "entity_location"

    return "description"


def evidence_proxy_hit(q_type: str, question: str, chunks: List[Dict[str, Any]]) -> bool:
    q = (question or "").lower()
    text_blob = "\n".join(str(c.get("text", "") or "") for c in chunks)
    meta_blob = " ".join(
        str(c.get("source_url", "") or "") + " " + str(c.get("domain", "") or "")
        for c in chunks
    )
    blob = (text_blob + "\n" + meta_blob).lower()

    if q_type == "event_date":
        return bool(MONTH_RE.search(blob) or YEAR_RE.search(blob) or TIME_RE.search(blob))

    if q_type == "numeric":
        return bool(DIGIT_RE.search(blob))

    if q_type == "list":
        if LIST_MARKER_RE.search(text_blob):
            return True
        # Fallback: list-like punctuation density.
        return text_blob.count(",") >= 3 or text_blob.count(";") >= 2

    if q_type == "entity_location":
        if any(t in q for t in WEBSITE_TERMS):
            return bool(DOMAIN_RE.search(blob))
        if any(t in q for t in LOCATION_TERMS) or "pittsburgh" in q:
            if "pittsburgh, pa" in blob:
                return True
            return bool(ADDRESS_RE.search(blob))
        # Generic entity-like proxy: at least one non-trivial chunk exists.
        return any(len((c.get("text") or "").strip()) > 60 for c in chunks)

    # description
    return any(len((c.get("text") or "").strip()) > 80 for c in chunks)


def stratified_sample(
    qids_by_type: Dict[str, List[str]],
    sample_size: int,
    seed: int = 13,
) -> List[str]:
    rng = random.Random(seed)
    all_ids = [qid for ids in qids_by_type.values() for qid in ids]
    n_total = len(all_ids)
    if sample_size >= n_total:
        out = all_ids[:]
        rng.shuffle(out)
        return out

    # Proportional allocation + largest remainder.
    alloc: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    for t, ids in qids_by_type.items():
        exact = sample_size * (len(ids) / max(1, n_total))
        base = int(exact)
        alloc[t] = min(base, len(ids))
        remainders.append((exact - base, t))

    used = sum(alloc.values())
    left = sample_size - used
    for _frac, t in sorted(remainders, reverse=True):
        if left <= 0:
            break
        if alloc[t] < len(qids_by_type[t]):
            alloc[t] += 1
            left -= 1

    # Ensure any non-empty type gets at least one slot when possible.
    non_empty_types = [t for t, ids in qids_by_type.items() if ids]
    if sample_size >= len(non_empty_types):
        for t in non_empty_types:
            if alloc[t] == 0:
                donor = max(non_empty_types, key=lambda x: alloc[x])
                if alloc[donor] > 1:
                    alloc[donor] -= 1
                    alloc[t] = 1

    sampled: List[str] = []
    for t, ids in qids_by_type.items():
        pool = ids[:]
        rng.shuffle(pool)
        sampled.extend(pool[: alloc.get(t, 0)])
    rng.shuffle(sampled)
    return sampled[:sample_size]


def mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def normalize_label(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"supported", "support", "s"}:
        return "supported"
    if s in {"unsupported", "not_supported", "not supported", "u", "hallucinated"}:
        return "unsupported"
    if s in {"idk", "i don't know", "dont know", "unknown"}:
        return "idk"
    return ""


def rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)

