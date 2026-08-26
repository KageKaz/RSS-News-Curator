#!/usr/bin/env python3
"""
Daily RSS curator.

Reads a list of RSS feed URLs, looks at everything published in the last
24 hours, collapses duplicate coverage of the same story, asks an LLM
(Groq's free API) to pick the 1-5 best articles, and writes a new RSS
feed containing just those picks.

Usage:
    GROQ_API_KEY=xxxx python curate_feed.py

Config is via environment variables (all optional except GROQ_API_KEY):
    GROQ_API_KEY      Required. Free key from https://console.groq.com
    FEEDS_FILE        Path to a text file of feed URLs. Default: feeds.txt
    OUTPUT_PATH       Where to write the curated feed. Default: docs/feed.xml
    OUTPUT_FEED_LINK  Public URL of the curated feed once hosted (used as
                       the feed's self-link). Default: a placeholder.
    LOOKBACK_HOURS    How far back to consider articles. Default: 24
    MAX_PICKS         Upper bound on articles selected. Default: 5
    MIN_PICKS         Lower bound on articles selected. Default: 1
    GROQ_MODEL        Model to use. Default: llama-3.3-70b-versatile
"""

import calendar
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import feedparser
import requests
from feedgen.feed import FeedGenerator

# ---- config -----------------------------------------------------------

FEEDS_FILE = os.environ.get("FEEDS_FILE", "feeds.txt")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "docs/feed.xml")
OUTPUT_FEED_LINK = os.environ.get(
    "OUTPUT_FEED_LINK", "https://example.github.io/rss-curator/feed.xml"
)
LOOKBACK_HOURS = float(os.environ.get("LOOKBACK_HOURS", "24"))
MAX_PICKS = int(os.environ.get("MAX_PICKS", "5"))
MIN_PICKS = int(os.environ.get("MIN_PICKS", "1"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEDUPE_THRESHOLD = float(os.environ.get("DEDUPE_THRESHOLD", "0.55"))

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


# ---- step 1: load feed list -------------------------------------------

def load_feed_urls(path):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


# ---- step 2: fetch + filter to last N hours ----------------------------

def clean_html(text):
    if not text:
        return ""
    text = TAG_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def entry_published_dt(entry):
    """Return a UTC datetime for an entry, or None if we can't tell."""
    struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if not struct:
        return None
    return datetime.fromtimestamp(calendar.timegm(struct), tz=timezone.utc)


def fetch_recent_entries(feed_urls, hours):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    entries = []
    for url in feed_urls:
        parsed = feedparser.parse(url)
        if parsed.bozo and not parsed.entries:
            print(f"  ! could not parse {url}: {parsed.bozo_exception}", file=sys.stderr)
            continue
        source_name = parsed.feed.get("title", url)
        for e in parsed.entries:
            pub = entry_published_dt(e)
            if pub is None or pub < cutoff:
                continue
            entries.append({
                "title": clean_html(e.get("title", "")).strip(),
                "link": e.get("link", ""),
                "summary": clean_html(e.get("summary", e.get("description", "")))[:600],
                "source": source_name,
                "published": pub,
            })
    return entries


# ---- step 3: dedupe / cluster same-story coverage ----------------------

def normalize_title(title):
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    return WS_RE.sub(" ", title).strip()


def title_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def cluster_entries(entries, threshold):
    """Greedy clustering: same story reported by multiple sources collapses
    into one candidate, keeping the earliest version and a count of how
    many sources covered it."""
    clusters = []  # each: {"rep": entry, "sources": set(), "count": int}
    for e in entries:
        norm = normalize_title(e["title"])
        placed = False
        for c in clusters:
            if title_similarity(norm, c["norm"]) >= threshold:
                c["sources"].add(e["source"])
                c["count"] += 1
                # keep the earliest-published version as the representative
                if e["published"] < c["rep"]["published"]:
                    c["rep"] = e
                placed = True
                break
        if not placed:
            clusters.append({
                "norm": norm,
                "rep": e,
                "sources": {e["source"]},
                "count": 1,
            })
    return clusters


# ---- step 4: ask Groq to pick the best ----------------------------------

def build_prompt(clusters):
    lines = [
        "You are a discerning news editor picking today's must-read articles "
        "from a pool of candidates gathered over the last 24 hours. Duplicate "
        "coverage of the same story has already been merged; the source count "
        "below tells you how many outlets covered it.",
        "",
        f"Select between {MIN_PICKS} and {MAX_PICKS} articles. Prioritize genuine "
        "significance and novelty over sensationalism, and prefer a diverse set "
        "of topics over several similar stories. It's fine to select fewer than "
        f"{MAX_PICKS} if the pool is thin.",
        "",
        "Candidates:",
    ]
    for i, c in enumerate(clusters):
        rep = c["rep"]
        lines.append(
            f"[{i}] \"{rep['title']}\" "
            f"(covered by {c['count']} source{'s' if c['count'] != 1 else ''}: "
            f"{', '.join(sorted(c['sources']))[:120]})\n"
            f"    {rep['summary'][:300]}"
        )
    lines.append("")
    lines.append(
        "Respond with ONLY a JSON object of the form "
        '{"picks": [{"index": <int>, "reason": "<one sentence>"}]} '
        "and nothing else."
    )
    return "\n".join(lines)


def call_groq(prompt):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    resp = requests.post(
        GROQ_ENDPOINT,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return parse_llm_json(content)


def parse_llm_json(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def select_best(clusters):
    if not clusters:
        return []
    prompt = build_prompt(clusters)
    data = call_groq(prompt)
    picks = data.get("picks", [])[:MAX_PICKS]
    selected = []
    for p in picks:
        idx = p.get("index")
        if idx is None or not (0 <= idx < len(clusters)):
            continue
        c = clusters[idx]
        selected.append({**c["rep"], "reason": p.get("reason", ""), "source_count": c["count"]})
    return selected


# ---- step 5: write the output feed --------------------------------------

def generate_feed(selected, output_path):
    fg = FeedGenerator()
    fg.title("Daily curated picks")
    fg.link(href=OUTPUT_FEED_LINK, rel="self")
    fg.description("Automatically curated best articles from the last 24 hours")
    fg.language("en")

    for item in selected:
        fe = fg.add_entry()
        fe.title(item["title"])
        fe.link(href=item["link"])
        desc = item.get("reason", "")
        if item.get("source_count", 1) > 1:
            desc += f" (covered by {item['source_count']} sources)"
        fe.description(desc or item.get("summary", ""))
        fe.guid(item["link"], permalink=True)
        fe.pubDate(item["published"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fg.rss_file(output_path)


# ---- main ----------------------------------------------------------------

def main():
    feed_urls = load_feed_urls(FEEDS_FILE)
    print(f"Loaded {len(feed_urls)} feed(s) from {FEEDS_FILE}")

    entries = fetch_recent_entries(feed_urls, LOOKBACK_HOURS)
    print(f"Found {len(entries)} article(s) in the last {LOOKBACK_HOURS:.0f}h")

    clusters = cluster_entries(entries, DEDUPE_THRESHOLD)
    print(f"Collapsed to {len(clusters)} distinct stor{'y' if len(clusters)==1 else 'ies'}")

    selected = select_best(clusters)
    print(f"Selected {len(selected)} article(s):")
    for s in selected:
        print(f"  - {s['title']}  ({s['link']})")

    generate_feed(selected, OUTPUT_PATH)
    print(f"Wrote curated feed to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
