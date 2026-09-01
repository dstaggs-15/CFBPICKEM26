"""
fetch_cfbd.py — pull REAL data from CollegeFootballData and land it.

Only file that talks to the outside world. Pulls three things:
  /games                 -> ids, teams, scores, neutral site, week, date
  /lines                 -> betting spread + over/under (the market benchmark)
  /stats/game/advanced   -> raw PPA / success rate / explosiveness per team-game

Saves raw parquet for each, plus a merged "base" table (games + market), and
prints a coverage report. Does NOT compute model features yet — that's the next
step. The point of THIS step: prove real data flows and see the betting-line
coverage that was silently zero last year.

Run (PowerShell), after setting your key for the session:
    $env:CFBD_API_KEY = "your_key_here"
    python fetch_cfbd.py --seasons 2023            # test one season first
    python fetch_cfbd.py --seasons 2005-2024       # then the full pull
"""

from __future__ import annotations
import os
import time
import argparse
import numpy as np
import pandas as pd
import requests

# Load CFBD_API_KEY from a local .env file if present (never committed).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # falls back to a real environment variable

CFBD_BASE = "https://api.collegefootballdata.com"
RAW_DIR = "data/raw"
DERIVED_DIR = "data/derived"


def _norm(k: str) -> str:
    return k.lower().replace("_", "").replace(".", "")

def _get_field(d: dict, *candidates, default=None):
    """Look up a key ignoring case / underscores / dots. Tries each candidate."""
    flat = {_norm(k): v for k, v in d.items()}
    for c in candidates:
        v = flat.get(_norm(c))
        if v is not None:
            return v
    return default


def _headers():
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise SystemExit(
            "CFBD_API_KEY is not set.\n"
            '  PowerShell:  $env:CFBD_API_KEY = "your_key_here"\n'
            "  then re-run this script."
        )
    return {"Authorization": f"Bearer {key}"}


def _get(path, **params):
    last_err = None
    for attempt in range(5):
        try:
            r = requests.get(f"{CFBD_BASE}{path}", headers=_headers(), params=params, timeout=120)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            wait = 3 * (attempt + 1)
            print(f"    ...{path} timed out/dropped (attempt {attempt+1}/5), retrying in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code >= 500:
            last_err = requests.exceptions.HTTPError(f"{r.status_code} server error")
            wait = 3 * (attempt + 1)
            print(f"    ...{path} returned {r.status_code} (attempt {attempt+1}/5), retrying in {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Gave up on {path} {params} after 5 attempts. Last error: {last_err}")


def parse_games(raw, season):
    rows = []
    for g in raw:
        stype = str(_get_field(g, "seasonType", "season_type", default="regular")).lower()
        rows.append({
            "game_id": str(_get_field(g, "id", "gameId")),
            "season": int(_get_field(g, "season", default=season)),
            "week": int(_get_field(g, "week", default=0)),
            "date": _get_field(g, "startDate", "start_date"),
            "home_team": _get_field(g, "homeTeam", "home_team"),
            "away_team": _get_field(g, "awayTeam", "away_team"),
            "neutral_site": bool(_get_field(g, "neutralSite", "neutral_site", default=False)),
            "home_points": _get_field(g, "homePoints", "home_points"),
            "away_points": _get_field(g, "awayPoints", "away_points"),
            "is_postseason": int(stype == "postseason"),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    for c in ("home_points", "away_points"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def parse_lines(raw):
    """Median spread + over/under across all providers for each game."""
    rows = []
    for g in raw:
        gid = str(_get_field(g, "id", "gameId"))
        for ln in _get_field(g, "lines", default=[]) or []:
            rows.append({
                "game_id": gid,
                "spread_home": pd.to_numeric(_get_field(ln, "spread"), errors="coerce"),
                "over_under": pd.to_numeric(_get_field(ln, "overUnder", "over_under"), errors="coerce"),
            })
    if not rows:
        return pd.DataFrame(columns=["game_id", "spread_home", "over_under"])
    df = pd.DataFrame(rows)
    return df.groupby("game_id", as_index=False)[["spread_home", "over_under"]].median()


def parse_advanced(raw):
    """Raw opponent-UNadjusted advanced stats per (game, team). Adjusted later."""
    rows = []
    for s in raw:
        off = _get_field(s, "offense", default={}) or {}
        deff = _get_field(s, "defense", default={}) or {}
        rows.append({
            "game_id": str(_get_field(s, "gameId", "game_id")),
            "team": _get_field(s, "team"),
            "off_ppa": pd.to_numeric(_get_field(off, "ppa"), errors="coerce"),
            "off_success": pd.to_numeric(_get_field(off, "successRate", "success_rate"), errors="coerce"),
            "off_explosive": pd.to_numeric(_get_field(off, "explosiveness"), errors="coerce"),
            "def_ppa": pd.to_numeric(_get_field(deff, "ppa"), errors="coerce"),
        })
    return pd.DataFrame(rows)


def implied_prob_from_spread(spread_home):
    return 1 / (1 + np.exp(-(-spread_home.astype(float)) * 0.14))


def _season_list(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2023",
                    help="e.g. '2023', '2005-2024', or '2019,2020,2021'")
    args = ap.parse_args()
    seasons = _season_list(args.seasons)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(DERIVED_DIR, exist_ok=True)

    games_all, lines_all, adv_all = [], [], []
    for yr in seasons:
        print(f"  fetching {yr} ...", flush=True)
        games_all.append(parse_games(_get("/games", year=yr, seasonType="both"), yr))
        lines_all.append(parse_lines(_get("/lines", year=yr)))
        adv_all.append(parse_advanced(_get("/stats/game/advanced", year=yr)))
        time.sleep(0.4)

    games = pd.concat(games_all, ignore_index=True)
    lines = pd.concat(lines_all, ignore_index=True)
    adv = pd.concat(adv_all, ignore_index=True)

    games.to_parquet(f"{RAW_DIR}/games_raw.parquet", index=False)
    lines.to_parquet(f"{RAW_DIR}/lines_raw.parquet", index=False)
    adv.to_parquet(f"{RAW_DIR}/advanced_raw.parquet", index=False)

    base = games.merge(lines, on="game_id", how="left")
    base["market_home_prob"] = implied_prob_from_spread(base["spread_home"])
    base.to_parquet(f"{DERIVED_DIR}/games_base.parquet", index=False)

    print("\n" + "=" * 52)
    print(f"FETCH COMPLETE — {len(base)} games, seasons {min(seasons)}-{max(seasons)}")
    print("=" * 52)
    def cov(col):
        return f"{base[col].notna().mean():.1%}" if col in base else "MISSING"
    print(f"  games with final score : {base['home_points'].notna().mean():.1%}")
    print(f"  spread_home coverage   : {cov('spread_home')}   <- was 0% last year")
    print(f"  over_under coverage    : {cov('over_under')}")
    print(f"  advanced-stat rows     : {len(adv)}  ({adv['off_ppa'].notna().mean():.1%} have PPA)")
    print(f"\n  saved: {DERIVED_DIR}/games_base.parquet (+ raw pieces in {RAW_DIR}/)")
    print("  next step: build features from these, then train + backtest.")


if __name__ == "__main__":
    main()
