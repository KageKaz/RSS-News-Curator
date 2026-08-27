# RSS curator

Every day, pulls your RSS feeds, keeps only what was published in the last
24 hours, merges duplicate coverage of the same story, and uses a free LLM
call (Groq) to pick the best articles. Publishes the result as a new
RSS feed anyone can subscribe to.

## Setup (about 10 minutes)

1. **Create a repo.** Push this folder to a new public GitHub repo (public
   repos get unlimited free GitHub Actions minutes).

2. **Get a free Groq API key.** Sign up at https://console.groq.com — no
   credit card required. Create an API key.

3. **Add the key as a repo secret.** In your repo: Settings → Secrets and
   variables → Actions → New repository secret.
   - Name: `GROQ_API_KEY`
   - Value: your key

4. **Edit `feeds.txt`** — replace the placeholder feeds with your own 10.

5. **Enable GitHub Pages.** Settings → Pages → Source: "Deploy from a
   branch" → Branch: `main`, folder: `/docs`. Save. GitHub will give you a
   URL like `https://yourname.github.io/your-repo/`.

6. **Set the feed's public URL** (optional but tidy). Settings → Secrets
   and variables → Actions → Variables tab → New repository variable:
   - Name: `OUTPUT_FEED_LINK`
   - Value: `https://yourname.github.io/your-repo/feed.xml`

7. **Run it once manually** to check everything works: Actions tab →
   "Daily RSS curation" → Run workflow. After it finishes, `docs/feed.xml`
   should appear in your repo, and after Pages finishes deploying (a
   minute or two) it'll be live at your Pages URL.

From then on it runs automatically once a day (default 13:00 UTC — change
the `cron` line in `.github/workflows/daily-curate.yml` to adjust). Point
any RSS reader at your Pages URL to subscribe to the curated feed.

## Tuning

All of these are environment variables you can set in the workflow file:

- `MAX_PICKS` / `MIN_PICKS` — how many articles to select (default 1–5)
- `LOOKBACK_HOURS` — the time window (default 24)
- `DEDUPE_THRESHOLD` — how aggressively similar titles get merged as the
  same story (0–1, default 0.55; higher = stricter matching, fewer merges)
- `GROQ_MODEL` — swap models if you want (default `openai/gpt-oss-120b`; check
  https://console.groq.com/docs/models for the current list if you get a
  `model_not_found` / 404 error, since Groq periodically deprecates models)

## Handling large numbers of articles (without bias toward verbose feeds)

Full summaries are stored, but before being shown to the model, every
candidate's summary is trimmed to the *same* length
(`SUMMARY_DISPLAY_CHARS`, default 220 characters) — regardless of how
verbose that particular feed's descriptions happen to be. This matters:
LLMs are prone to "verbosity bias," rating longer, more detailed-looking
text as more substantive, even when the actual newsworthiness is
identical. If one candidate showed up with a 3000-character summary and
another with a 40-character one, the model would tend to favor the
longer one for reasons that have nothing to do with which story actually
matters more. Capping everyone to the same length removes that skew,
and the prompt also explicitly tells the model not to treat summary
length as a signal of importance. A shorter feed still shows its
(genuinely shorter) summary in full — nobody is padded up, just nobody
gets more than anyone else.

On top of that, batches are sized dynamically by *estimated token count*
rather than a fixed article count, so this scales automatically whether
there are 15 articles or 1,500:

- If everything fits in one batch, it's a single API call — same as
  before.
- If not, each batch shortlists its most promising few stories, then one
  final call picks from the combined shortlist. Batches run with a short
  pause between them so consecutive calls don't add up past Groq's
  per-minute token limit.
- Tested up to 1,500 distinct stories: batching itself is near-instant,
  no articles are silently dropped, and runtime scales roughly linearly
  at about 15 seconds per batch (a very large day might mean a few
  minutes total — trivial for a once-a-day job).

Tuning knobs:

- `SUMMARY_DISPLAY_CHARS` — the equal length cap applied to every
  candidate's summary before it's shown to the model (default 220).
  Raise it for more context per story (at the cost of fewer candidates
  per batch); the equal-treatment property holds at any value.
- `BATCH_TOKEN_BUDGET` — target tokens per batch (default 6000, leaving
  headroom under Groq's 8,000 tokens-per-minute limit for
  `openai/gpt-oss-120b`). Lower this if you ever see a `413 Payload Too
  Large` or a 429 rate-limit error.
- `MAX_CANDIDATES_PER_BATCH` — a secondary cap (default 60) so a batch
  doesn't end up with so many candidates that the model's attention gets
  diluted, even if there'd be token room for more.
- `RAW_SUMMARY_CHARS_CAP` — a safety net (default 3000 characters) on how
  much of each article's summary is stored at fetch time, purely to stop
  one feed with a pathologically long "summary" field (e.g. a full-text
  feed embedding whole articles) from causing problems downstream. This
  doesn't affect normal RSS summaries, which are almost always well
  under it.
- `PICKS_PER_BATCH` — how many stories each batch shortlists (default 8)
- `BATCH_SLEEP_SECONDS` — pause between calls (default 60s)

Note: with a wide/diverse candidate pool, `DEDUPE_THRESHOLD` may need
tuning too — headlines that share a lot of common phrasing (e.g. "X hits
Y region") can get merged as duplicates even when they're different
stories. Lower the threshold if you see genuine stories disappearing, or
check the printed "distinct stories" count in the script's output against
how many you'd expect.

## Local testing

```
pip install -r requirements.txt
GROQ_API_KEY=your_key python curate_feed.py
```

This writes `docs/feed.xml` locally so you can check it before relying on
the scheduled run.

## Notes on cost

Groq's free tier (no card needed) covers this easily — one call a day
against a handful of headlines is a tiny fraction of the free daily quota.
If you ever want to swap in a different provider (Gemini, or Claude via
the Anthropic API), only `call_groq()` needs to change — everything else
stays the same.