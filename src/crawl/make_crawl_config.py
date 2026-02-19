#!/usr/bin/env python3
"""
Generate a crawl configuration file for crawl.py.

Outputs:
  configs/crawl_config.yaml

This config is intentionally focused (not a general web crawler):
- strong allow/deny regexes per domain
- conservative max_depth / max_pages
- throttling
- special-cases for event calendar domains
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml 


def domain_cfg(
    domain: str,
    seeds: List[str],
    *,
    out_subdir: str,
    max_pages: int,
    max_depth: int,
    delay_s: float,
    allow_patterns: List[str],
    deny_patterns: List[str],
    allowed_schemes: List[str] = ["http", "https"],
    same_domain_only: bool = True,
    strip_query: bool = False,
    keep_query_params: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "domain": domain,
        "seeds": seeds,
        "out_subdir": out_subdir,
        "max_pages": max_pages,
        "max_depth": max_depth,
        "delay_s": delay_s,
        "allowed_schemes": allowed_schemes,
        "same_domain_only": same_domain_only,
        "strip_query": strip_query,
        "keep_query_params": keep_query_params or [],
        "allow_patterns": allow_patterns,
        "deny_patterns": deny_patterns,
    }


def main() -> None:
    config: Dict[str, Any] = {
        "version": 1,
        "user_agent": "Mozilla/5.0 (compatible; ANLP-RAG-Scraper/1.0; +https://cmu.edu)",
        "out_dir": "data/raw",
        "manifest_path": "data/raw/manifest.jsonl",
        "global_deny_patterns": [
            r"/search", r"/login", r"/signin", r"/signup", r"/account",
            r"/cart", r"/checkout", r"/wp-admin",
            r"\.jpg$|\.jpeg$|\.png$|\.gif$|\.webp$|\.svg$",
            r"\.mp4$|\.mov$|\.avi$|\.mp3$|\.wav$",
            r"\.css$|\.js$|\.map$",
        ],
        "domains": [],
    }

    # -------- General info & history --------
    config["domains"].append(domain_cfg(
        "en.wikipedia.org",
        seeds=[
            "https://en.wikipedia.org/wiki/Pittsburgh",
            "https://en.wikipedia.org/wiki/History_of_Pittsburgh",
        ],
        out_subdir="wikipedia",
        max_pages=200,
        max_depth=1,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://en\.wikipedia\.org/wiki/(Pittsburgh|History_of_Pittsburgh)",
            # Other reelevant wiki pages
            r"^https?://en\.wikipedia\.org/wiki/(Carnegie_Museum|Carnegie_Museums|Heinz_History_Center|Pittsburgh_Symphony|Pittsburgh_Penguins|Pittsburgh_Steelers|Pittsburgh_Pirates)",
        ],
        deny_patterns=[
            r":",  # exclude Special:, File:, Category:, Talk:, etc.
            r"#",  # fragments
        ],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "www.britannica.com",
        seeds=["https://www.britannica.com/place/Pittsburgh"],
        out_subdir="britannica",
        max_pages=60,
        max_depth=1,
        delay_s=1.2,
        allow_patterns=[
            r"^https?://www\.britannica\.com/place/Pittsburgh",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "www.cmu.edu",
        seeds=["https://www.cmu.edu/about/"],
        out_subdir="cmu",
        max_pages=300,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://www\.cmu\.edu/about",
            r"^https?://www\.cmu\.edu/about/history",
            r"^https?://www\.cmu\.edu/about/.*(facts|leadership|campus|rankings|mission|values)",
            # events sources are also on cmu.edu but we’ll handle events.cmu.edu separately
            r"^https?://www\.cmu\.edu/engage/alumni/events/campus",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "www.visitpittsburgh.com",
        seeds=[
            "https://www.visitpittsburgh.com/",
            "https://www.visitpittsburgh.com/things-to-do/pittsburgh-sports-teams/",
            "https://www.visitpittsburgh.com/events-festivals/food-festivals/",
        ],
        out_subdir="visitpittsburgh",
        max_pages=2000,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://www\.visitpittsburgh\.com/(things-to-do|events-festivals|food|restaurants|sports|music|arts-culture|history|plan-your-trip|neighborhoods)",
            r"^https?://www\.visitpittsburgh\.com/events-festivals/",
            r"^https?://www\.visitpittsburgh\.com/things-to-do/",
        ],
        deny_patterns=[
            r"\?.*(utm_|fbclid|gclid)=",
            r"#",
        ],
        strip_query=True,
    ))

    # City of Pittsburgh (HTML + PDFs)
    config["domains"].append(domain_cfg(
        "www.pittsburghpa.gov",
        seeds=[
            "https://www.pittsburghpa.gov/Home",
            "https://pittsburghpa.gov/finance/tax-forms",
            # budget pdf
            "https://www.pittsburghpa.gov/files/assets/city/v/4/omb/documents/operating-budgets/2025-operating-budget.pdf",
        ],
        out_subdir="pittsburghpa",
        max_pages=800,
        max_depth=2,
        delay_s=1.2,
        allow_patterns=[
            r"^https?://(www\.)?pittsburghpa\.gov/(Home|finance|files|Government|Residents|Business|.*operating-budgets)",
            r"^https?://(www\.)?pittsburghpa\.gov/finance/tax-forms",
            r"\.pdf$",
        ],
        deny_patterns=[
            r"#",
            r"\?.*(utm_|fbclid|gclid)=",
        ],
        strip_query=True,
    ))

    # -------- Events --------
    # pittsburgh events calendar
    config["domains"].append(domain_cfg(
        "pittsburgh.events",
        seeds=[
            "https://pittsburgh.events/",
            "https://pittsburgh.events/march/",
            "https://pittsburgh.events/april/",
            "https://pittsburgh.events/may/",
            "https://pittsburgh.events/june/",
            "https://pittsburgh.events/july/",
            "https://pittsburgh.events/august/",
            "https://pittsburgh.events/september/",
            "https://pittsburgh.events/october/",
            "https://pittsburgh.events/november/",
            "https://pittsburgh.events/december/",

        ],
        out_subdir="pittsburgh_events",
        max_pages=2000,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://pittsburgh\.events/",
            # common patterns: /events/..., /event/..., /calendar/..., /month/...
            r"^https?://pittsburgh\.events/(events|event|calendar|month|search|.*\d{4}.*\d{2})",
        ],
        deny_patterns=[
            r"#",
            r"\?.*(utm_|fbclid|gclid)=",
            # avoid deep paging / infinite scroll variants if present
            r"(page=|p=)\d{3,}",
        ],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "downtownpittsburgh.com",
        seeds=["https://downtownpittsburgh.com/events/"],
        out_subdir="downtownpgh_events",
        max_pages=600,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://downtownpittsburgh\.com/events",
            r"^https?://downtownpittsburgh\.com/.*(event|events)/",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "www.pghcitypaper.com",
        seeds=["https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d"],
        out_subdir="pghcitypaper_events",
        max_pages=1200,
        max_depth=2,
        delay_s=1.2,
        allow_patterns=[
            r"^https?://www\.pghcitypaper\.com/pittsburgh/EventSearch",
            r"^https?://www\.pghcitypaper\.com/pittsburgh/.*(EventSearch|event|events|Event)",
        ],
        deny_patterns=[r"#"],
        strip_query=False,  # keep query for v=d and date filters
    ))

    # CMU events calendar is on events.cmu.edu
    config["domains"].append(domain_cfg(
        "events.cmu.edu",
        seeds=["https://events.cmu.edu/"],
        out_subdir="cmu_events",
        max_pages=1200,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://events\.cmu\.edu/",
            r"^https?://events\.cmu\.edu/.*(event|events|calendar|search)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "www.cmu.edu/engage/events",
        seeds=["https://www.cmu.edu/engage/events"],
        out_subdir="cmu_events",
        max_pages=1200,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://www\.cmu\.edu/engage/events",
            r"^https?://www\.cmu\.edu/engage/events/.*",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))
    # -------- Music & Culture --------
    config["domains"].append(domain_cfg(
        "www.pittsburghsymphony.org",
        seeds=["https://www.pittsburghsymphony.org/"],
        out_subdir="pso",
        max_pages=800,
        max_depth=2,
        delay_s=1.2,
        allow_patterns=[
            r"^https?://www\.pittsburghsymphony\.org/",
            r"^https?://www\.pittsburghsymphony\.org/.*(concerts|events|tickets|season|calendar|about|education)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "pittsburghopera.org",
        seeds=["https://pittsburghopera.org/"],
        out_subdir="opera",
        max_pages=600,
        max_depth=2,
        delay_s=1.2,
        allow_patterns=[
            r"^https?://pittsburghopera\.org/",
            r"^https?://pittsburghopera\.org/.*(performances|events|calendar|about|education|season)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "trustarts.org",
        seeds=["https://trustarts.org/"],
        out_subdir="trustarts",
        max_pages=1200,
        max_depth=2,
        delay_s=1.2,
        allow_patterns=[
            r"^https?://trustarts\.org/",
            r"^https?://trustarts\.org/.*(events|calendar|show|performance|theater|about)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "carnegiemuseums.org",
        seeds=["https://carnegiemuseums.org/"],
        out_subdir="carnegie_museums",
        max_pages=800,
        max_depth=2,
        delay_s=1.2,
        allow_patterns=[
            r"^https?://carnegiemuseums\.org/",
            r"^https?://carnegiemuseums\.org/.*(visit|exhibitions|events|about|museum|plan)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "www.heinzhistorycenter.org",
        seeds=["https://www.heinzhistorycenter.org/"],
        out_subdir="heinz_history",
        max_pages=600,
        max_depth=2,
        delay_s=1.2,
        allow_patterns=[
            r"^https?://www\.heinzhistorycenter\.org/",
            r"^https?://www\.heinzhistorycenter\.org/.*(visit|events|exhibits|about|education|calendar)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "www.thefrickpittsburgh.org",
        seeds=["https://www.thefrickpittsburgh.org/"],
        out_subdir="frick",
        max_pages=600,
        max_depth=2,
        delay_s=1.2,
        allow_patterns=[
            r"^https?://www\.thefrickpittsburgh\.org/",
            r"^https?://www\.thefrickpittsburgh\.org/.*(visit|events|exhibitions|about|education)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    # -------- Food festivals --------
    config["domains"].append(domain_cfg(
        "www.picklesburgh.com",
        seeds=["https://www.picklesburgh.com/"],
        out_subdir="picklesburgh",
        max_pages=300,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://www\.picklesburgh\.com/",
            r"^https?://www\.picklesburgh\.com/.*(about|festival|schedule|vendors|faq|music|map)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "www.pghtacofest.com",
        seeds=["https://www.pghtacofest.com/"],
        out_subdir="tacofest",
        max_pages=300,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://www\.pghtacofest\.com/",
            r"^https?://www\.pghtacofest\.com/.*(about|festival|schedule|vendors|faq|music|map)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "pittsburghrestaurantweek.com",
        seeds=["https://pittsburghrestaurantweek.com/"],
        out_subdir="restaurant_week",
        max_pages=500,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://pittsburghrestaurantweek\.com/",
            r"^https?://pittsburghrestaurantweek\.com/.*(participating|restaurants|events|about|faq|schedule)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "littleitalydays.com",
        seeds=["https://littleitalydays.com/"],
        out_subdir="little_italy_days",
        max_pages=300,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://littleitalydays\.com/",
            r"^https?://littleitalydays\.com/.*(festival|schedule|vendors|about|faq|music|map)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "bananasplitfest.com",
        seeds=["https://bananasplitfest.com/"],
        out_subdir="banana_split_fest",
        max_pages=300,
        max_depth=2,
        delay_s=1.0,
        allow_patterns=[
            r"^https?://bananasplitfest\.com/",
            r"^https?://bananasplitfest\.com/.*(festival|schedule|vendors|about|faq|music|map)",
        ],
        deny_patterns=[r"#"],
        strip_query=True,
    ))

    # -------- Sports  --------
    config["domains"].append(domain_cfg(
        "www.mlb.com",
        seeds=["https://www.mlb.com/pirates"],
        out_subdir="pirates",
        max_pages=120,
        max_depth=1,
        delay_s=1.5,
        allow_patterns=[
            r"^https?://www\.mlb\.com/pirates",
            r"^https?://www\.mlb\.com/pirates/(ballpark|history|team|fans|schedule)",
        ],
        deny_patterns=[
            r"/news", r"/video", r"/scores", r"/stats", r"#"
        ],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "www.steelers.com",
        seeds=["https://www.steelers.com/"],
        out_subdir="steelers",
        max_pages=200,
        max_depth=1,
        delay_s=1.5,
        allow_patterns=[
            r"^https?://www\.steelers\.com/",
            r"^https?://www\.steelers\.com/(team|history|stadium|tickets|schedule)",
        ],
        deny_patterns=[
            r"/news", r"/video", r"/photos", r"/podcasts", r"#"
        ],
        strip_query=True,
    ))

    config["domains"].append(domain_cfg(
        "www.nhl.com",
        seeds=["https://www.nhl.com/penguins"],
        out_subdir="penguins",
        max_pages=200,
        max_depth=1,
        delay_s=1.5,
        allow_patterns=[
            r"^https?://www\.nhl\.com/penguins",
            r"^https?://www\.nhl\.com/penguins/(team|history|arena|schedule|tickets)",
        ],
        deny_patterns=[
            r"/news", r"/video", r"/multimedia", r"#"
        ],
        strip_query=True,
    ))

    out_path = Path("configs/crawl_config.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    print(f"Wrote {out_path} with {len(config['domains'])} domain configs")


if __name__ == "__main__":
    main()
