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
    GROQ_MODEL        Model to use. Default: openai/gpt-oss-120b

Full article summaries are stored, but a consistent length cap is applied
before showing them to the model (to avoid biasing selection toward
whichever source happens to write longer summaries), and batches are
sized dynamically by estimated token count rather than a fixed article
count, so this scales automatically whether you have 10 articles or
10,000 — see BATCH_TOKEN_BUDGET and SUMMARY_DISPLAY_CHARS below.
"""

import calendar
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import feedparser
import requests
from feedgen.feed import FeedGenerator

# ---- config -----------------------------------------------------------

FEEDS_FILE = os.environ.get("FEEDS_FILE", "feeds.txt")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "docs/feed.xml")
OUTPUT_FEED_LINK = os.environ.get(
    "OUTPUT_FEED_LINK", "https://kagekaz.github.io/RSS-News-Curator/feed.xml"
)
LOOKBACK_HOURS = float(os.environ.get("LOOKBACK_HOURS", "24"))
MAX_PICKS = int(os.environ.get("MAX_PICKS", "5"))
MIN_PICKS = int(os.environ.get("MIN_PICKS", "1"))
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEDUPE_THRESHOLD = float(os.environ.get("DEDUPE_THRESHOLD", "0.5"))

# Full summaries are stored (no per-article truncation) other than this
# safety net, which only guards against a pathological feed embedding an
# entire multi-thousand-word article in its "summary" field.
RAW_SUMMARY_CHARS_CAP = int(os.environ.get("RAW_SUMMARY_CHARS_CAP", "3000"))

# What actually gets shown to the model, though, is capped to the SAME
# length for every candidate. This matters: feeds vary a lot in how
# verbose their summaries are, and that's a property of the feed's
# writing style, not the story's importance. If one candidate showed up
# with 3000 characters of summary and another with 40, the model would be
# prone to "verbosity bias" -- rating the longer one as more substantive
# just because there's more text, regardless of actual content. Capping
# everyone to the same length removes that confound while still giving
# far more context than a single short snippet would.
SUMMARY_DISPLAY_CHARS = int(os.environ.get("SUMMARY_DISPLAY_CHARS", "180"))

# Batches are packed by *estimated token count*, not a fixed article count,
# so this scales automatically regardless of how long summaries are or how
# many articles come in. BATCH_TOKEN_BUDGET leaves headroom under Groq's
# free-tier 8,000-tokens-per-minute limit for openai/gpt-oss-120b (room for
# the model's reply plus estimation slack). MAX_CANDIDATES_PER_BATCH is a
# secondary cap so a batch of very short summaries doesn't end up with so
# many candidates that the model's attention gets diluted.
BATCH_TOKEN_BUDGET = int(os.environ.get("BATCH_TOKEN_BUDGET", "6000"))
MAX_CANDIDATES_PER_BATCH = int(os.environ.get("MAX_CANDIDATES_PER_BATCH", "50"))
CHARS_PER_TOKEN = 3.5  # conservative estimate, errs toward smaller batches

# If there's more than one batch, each batch shortlists its most promising
# stories, then one final call picks from the combined shortlist.
PICKS_PER_BATCH = int(os.environ.get("PICKS_PER_BATCH", "4"))
BATCH_SLEEP_SECONDS = float(os.environ.get("BATCH_SLEEP_SECONDS", "60"))

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
                "title": clean_html(e.get("title", "")).strip()[:300],
                "link": e.get("link", ""),
                "summary": clean_html(e.get("summary", e.get("description", "")))[:RAW_SUMMARY_CHARS_CAP],
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

def rank_clusters(clusters):
    """Order candidates by how likely they are to matter: stories covered
    by more sources first (a decent proxy for significance), then most
    recent. Batches are filled in this order, so the most promising
    stories always land in the earliest (and therefore never-dropped)
    batches."""
    return sorted(clusters, key=lambda c: (c["count"], c["rep"]["published"]), reverse=True)


def estimate_tokens(text):
    return len(text) / CHARS_PER_TOKEN


def format_candidate_line(idx, c):
    rep = c["rep"]
    return (
        f"[{idx}] \"{rep['title']}\" "
        f"(covered by {c['count']} source{'s' if c['count'] != 1 else ''}: "
        f"{', '.join(sorted(c['sources']))[:120]})\n"
        f"    {rep['summary'][:SUMMARY_DISPLAY_CHARS]}"
    )


BOILERPLATE_TOKEN_ESTIMATE = 220  # rough size of the instructions text


def build_batches(ranked_clusters, token_budget, max_per_batch):
    """Pack candidates into batches sized by estimated token count (not a
    fixed article count), so this scales automatically whether summaries
    are short or full-length, and whether there are 10 articles or 10,000."""
    batches = []
    current = []
    current_tokens = BOILERPLATE_TOKEN_ESTIMATE

    for c in ranked_clusters:
        line = format_candidate_line(len(current), c)
        line_tokens = estimate_tokens(line)

        # Safety valve: clip a single pathologically long summary rather
        # than letting it blow the whole batch's budget on its own.
        room = token_budget - BOILERPLATE_TOKEN_ESTIMATE
        if line_tokens > room:
            max_chars = max(int(room * CHARS_PER_TOKEN) - 200, 0)
            c = {**c, "rep": {**c["rep"], "summary": c["rep"]["summary"][:max_chars]}}
            line = format_candidate_line(len(current), c)
            line_tokens = estimate_tokens(line)

        would_exceed_tokens = current and (current_tokens + line_tokens > token_budget)
        would_exceed_count = len(current) >= max_per_batch
        if would_exceed_tokens or would_exceed_count:
            batches.append(current)
            current = []
            current_tokens = BOILERPLATE_TOKEN_ESTIMATE

        current.append(c)
        current_tokens += line_tokens

    if current:
        batches.append(current)
    return batches


def build_prompt(clusters, max_picks, min_picks, include_reason=True):
    lines = [
        "You are a discerning news editor picking today's must-read articles "
        "from a pool of candidates gathered over the last 24 hours. Duplicate "
        "coverage of the same story has already been merged; the source count "
        "below tells you how many outlets covered it.",
        "",
        "Summaries are trimmed to a consistent length for every candidate, the first 180 characters. ",
        "",
        f"Select between {min_picks} and {max_picks} articles. Prioritize "
        "highly signficant articles, ones that would actually change/refine the reader's understanding of the world and global events. Cater to a California audience but keep them updated on relevant global news."
        "Give a moderate bias towards articles about education, and a slight bias to articles that concern the US. If the article is a political one, it must be something every American should know about. Avoid sensationalism."
        "It's fine to select fewer than "
        f"{max_picks} if the pool is thin.",
        "",
        "Candidates:",
    ]
    for i, c in enumerate(clusters):
        lines.append(format_candidate_line(i, c))
    lines.append("")
    if include_reason:
        lines.append(
            "Respond with ONLY a JSON object of the form "
            '{"picks": [{"index": <int>, "reason": "<one sentence>"}]} '
            "and nothing else."
        )
    else:
        lines.append(
            "Respond with ONLY a JSON object of the form "
            '{"picks": [{"index": <int>}]} and nothing else.'
        )
    return "\n".join(lines)


def call_groq(prompt, max_retries=3):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    for attempt in range(max_retries + 1):
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
        if resp.status_code == 429 and attempt < max_retries:
            wait = float(resp.headers.get("retry-after", 2 ** attempt * 10))
            print(f"  ! rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(wait)
            continue
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


def apply_picks(clusters, data, max_picks):
    picks = data.get("picks", [])[:max_picks]
    selected = []
    for p in picks:
        idx = p.get("index")
        if idx is None or not (0 <= idx < len(clusters)):
            continue
        c = clusters[idx]
        selected.append({**c["rep"], "reason": p.get("reason", ""), "source_count": c["count"]})
    return selected


def select_best(clusters):
    if not clusters:
        return []

    ranked = rank_clusters(clusters)
    batches = build_batches(ranked, BATCH_TOKEN_BUDGET, MAX_CANDIDATES_PER_BATCH)

    if len(batches) == 1:
        prompt = build_prompt(batches[0], MAX_PICKS, MIN_PICKS)
        data = call_groq(prompt)
        return apply_picks(batches[0], data, MAX_PICKS)

    # Too many/too long candidates for one call: shortlist each batch, then
    # make a final pick across the combined shortlist.
    print(f"  {len(clusters)} distinct stories -> {len(batches)} batches "
          f"(sizes: {[len(b) for b in batches]})")

    shortlist = []
    for i, batch in enumerate(batches):
        prompt = build_prompt(batch, PICKS_PER_BATCH, 1, include_reason=False)
        data = call_groq(prompt)
        picks = data.get("picks", [])[:PICKS_PER_BATCH]
        for p in picks:
            idx = p.get("index")
            if idx is not None and 0 <= idx < len(batch):
                shortlist.append(batch[idx])
        print(f"  batch {i + 1}/{len(batches)}: shortlisted {len(picks)}")
        if i < len(batches) - 1:
            time.sleep(BATCH_SLEEP_SECONDS)  # stay under the per-minute token limit

    if not shortlist:
        return []

    # The shortlist itself might still be too big/long for one call
    # (e.g. many batches each contributing 8 picks with full summaries),
    # so run it through the same batching logic rather than assuming.
    shortlist_ranked = rank_clusters(shortlist)
    final_batches = build_batches(shortlist_ranked, BATCH_TOKEN_BUDGET, MAX_CANDIDATES_PER_BATCH)
    if len(final_batches) > 1:
        # Rare, but if it happens, just take the top slice by rank rather
        # than recursing into another full shortlist round.
        final_candidates = shortlist_ranked[: max(len(b) for b in final_batches)]
    else:
        final_candidates = final_batches[0]

    time.sleep(BATCH_SLEEP_SECONDS)
    final_prompt = build_prompt(final_candidates, MAX_PICKS, MIN_PICKS)
    data = call_groq(final_prompt)
    return apply_picks(final_candidates, data, MAX_PICKS)


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