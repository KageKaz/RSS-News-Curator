# RSS curator

Every day, pulls your RSS feeds, keeps only what was published in the last
24 hours, merges duplicate coverage of the same story, and uses a free LLM
call (Groq) to pick the 1-5 best articles. Publishes the result as a new
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
- `GROQ_MODEL` — swap models if you want (default `llama-3.3-70b-versatile`)

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
