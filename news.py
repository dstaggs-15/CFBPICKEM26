"""
news.py — pull recent headlines per team for this week's slate.

Runs inside the weekly GitHub Action (the site is static, so it can't fetch news
live on click — it bakes headlines into docs/news.json at build time, and the
page shows what was gathered).

Design choice: we read Google News' RSS feed rather than scraping a site's HTML.
An RSS feed is a stable, public, machine-readable endpoint — far sturdier than
scraping a page whose layout changes and which may block bots. Same idea as
"scrape the news," done the durable way.

Output: docs/news.json  ->  {"teams": {"LSU": [{title,url,source}], ...}}
"""

from __future__ import annotations
import json
import time
import urllib.parse
from pathlib import Path

import requests

try:
    import feedparser
except ImportError:  # keep the pipeline alive even if the dep is missing
    feedparser = None

from weekly_input import load_slate

OUT_JSON = "docs/news.json"
PER_TEAM = 3
RSS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def team_headlines(team: str) -> list[dict]:
    if feedparser is None:
        return []
    q = urllib.parse.quote_plus(f'{team} college football')
    url = RSS.format(q=q)
    try:
        raw = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        raw.raise_for_status()
        feed = feedparser.parse(raw.content)
    except Exception as e:
        print(f"  news fetch failed for {team}: {e}")
        return []

    items = []
    for entry in feed.entries[:PER_TEAM]:
        title = getattr(entry, "title", "").strip()
        # Google News titles end with ' - Source'; split it out
        source = ""
        if " - " in title:
            title, source = title.rsplit(" - ", 1)
        items.append({
            "title": title,
            "url": getattr(entry, "link", ""),
            "source": source or getattr(getattr(entry, "source", None), "title", ""),
        })
    return items


def main():
    slate = load_slate()
    teams = []
    for m in slate:
        for t in (m.away_team, m.home_team):
            if t not in teams:
                teams.append(t)

    out = {}
    for t in teams:
        out[t] = team_headlines(t)
        time.sleep(0.5)  # be gentle
        print(f"  {t}: {len(out[t])} headlines")

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"teams": out}, f, indent=2)
    print(f"Wrote {OUT_JSON} for {len(teams)} teams.")


if __name__ == "__main__":
    main()
