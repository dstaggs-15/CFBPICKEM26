"""
fetch_cfbd.py — where YOUR CFBD key plugs in.

This is the ONLY place that talks to the outside world. Its single job: pull raw
CFBD data and return the canonical games table that schema.py defines. Nothing
downstream knows or cares that CFBD exists — swap the source here and the whole
pipeline still works.

The non-negotiable pattern, and the whole point of the rebuild:

    fetch  ->  map to schema  ->  contract.validate()  ->  save

If the mapping is wrong (a stat endpoint changed shape, a join misses), the
contract raises BEFORE anything is saved or trained. You find out on fetch day,
not on game day.

Set CFBD_API_KEY in your environment / GitHub Actions secret. This file cannot
reach the CFBD API from Anthropic's sandbox, so it's written to run on YOUR
machine or in the Action.
"""

from __future__ import annotations
import os
import sys
import requests
import pandas as pd

import contract, schema

CFBD_BASE = "https://api.collegefootballdata.com"
OUT_PARQUET = "data/derived/games.parquet"


def _headers():
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise SystemExit("Set CFBD_API_KEY (env var or GitHub secret) before fetching.")
    return {"Authorization": f"Bearer {key}"}


def _get(path, **params):
    r = requests.get(f"{CFBD_BASE}{path}", headers=_headers(), params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def build_games_table(start_season: int, end_season: int) -> pd.DataFrame:
    """
    TODO(you): fill in the mapping from CFBD endpoints to the canonical columns.
    The skeleton below shows the shape and the endpoints you'll use. Each mapped
    block should end up populating the schema columns; the contract will tell you
    precisely which ones you missed.

    Endpoints you'll draw on:
      /games                     -> ids, teams, scores, neutral_site, week, date
      /lines                     -> spread_home, over_under  (market benchmark)
      /stats/game/advanced       -> ppa / success_rate / explosiveness (raw)
      /ratings/... or your Elo   -> elo_home_prob
    The opponent-ADJUSTED strength diffs are computed in a features step (next
    layer) from the raw advanced stats — not here. Here we just land raw data +
    context + market, and elo_home_prob from the EloModel.
    """
    frames = {}
    for season in range(start_season, end_season + 1):
        games = pd.DataFrame(_get("/games", year=season, seasonType="both"))
        frames[season] = games
        # ... map lines, advanced stats, etc. (left for the wiring step)

    raise NotImplementedError(
        "Fill in build_games_table(): map CFBD JSON -> schema.ALL_COLS, then let "
        "the contract check it. Run scripts/demo.py meanwhile to work against "
        "synthetic data."
    )


def main(start_season=2005, end_season=2024):
    df = build_games_table(start_season, end_season)

    # Land raw context/market coverage first; strength features get filled in the
    # features step, so at fetch time we validate the columns we DO own here.
    contract.validate(
        df, require_market=True, stage="training",
        strength_min_coverage=0.0,  # strength diffs computed later, not here
    )

    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {len(df)} games to {OUT_PARQUET}")
    print(contract.report(df))


if __name__ == "__main__":
    main()
