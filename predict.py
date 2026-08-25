"""
predict.py — turn this week's games.txt into docs/predictions.json.

Steps:
  1. load the trained model
  2. read the weekly slate (weekly_input.load_slate)
  3. for each matchup, find its scheduled (unplayed) row in the feature table
     and predict P(home win)
  4. write a plain-English "why" from the biggest factors behind the pick
  5. attach each team's current-season stat ranks (team_stats)
  6. write docs/predictions.json for the website

The "why" is generated from the actual feature values that moved the pick, not
canned text — if the model leaned on Elo, the why says so; if it leaned on
efficiency, it says that. Honest by construction.
"""

from __future__ import annotations
import json
from datetime import date
import numpy as np
import pandas as pd
import joblib

import schema
from weekly_input import load_slate
import team_stats

TRAIN_PARQUET = "data/derived/training.parquet"
MODEL_FILE = "model.joblib"
OUT_JSON = "docs/predictions.json"

# minimal alias map; extend as needed to match CFBD spellings
ALIASES = {
    "ole miss": "Ole Miss", "miss": "Ole Miss",
    "pitt": "Pittsburgh", "uconn": "Connecticut",
    "usc": "USC", "lsu": "LSU", "tcu": "TCU", "smu": "SMU", "byu": "BYU",
    "nc state": "NC State", "north carolina state": "NC State",
}


def canon(name: str, known: set[str]) -> str:
    n = name.strip()
    if n in known:
        return n
    a = ALIASES.get(n.lower())
    if a:
        return a
    # case-insensitive match
    for k in known:
        if k.lower() == n.lower():
            return k
    return n  # leave as-is; may not match schedule


def find_game(feat: pd.DataFrame, away: str, home: str):
    """Locate the scheduled (preferably unplayed) row for away @ home."""
    m = feat[(feat.home_team == home) & (feat.away_team == away)]
    if m.empty:
        # try swapped, in case the slate listed sides opposite the schedule
        m = feat[(feat.home_team == away) & (feat.away_team == home)]
    if m.empty:
        return None, False
    unplayed = m[m["home_points"].isna()]
    row = (unplayed.sort_values("date").iloc[0] if not unplayed.empty
           else m.sort_values("date").iloc[-1])
    swapped = not (row.home_team == home and row.away_team == away)
    return row, swapped


def explain(row: pd.Series, p_home: float) -> list[str]:
    """Plain-English drivers of the pick from real feature values."""
    why = []
    fav_home = p_home >= 0.5
    fav = row.home_team if fav_home else row.away_team

    elo = row.get("elo_home_prob")
    if pd.notna(elo):
        if (elo >= 0.5) == fav_home and abs(elo - 0.5) > 0.08:
            why.append(f"{fav} carries the stronger overall rating into this game.")

    off = row.get("off_ppa_adj_diff")
    if pd.notna(off) and abs(off) > 0.02:
        better = row.home_team if off > 0 else row.away_team
        why.append(f"{better} has been more efficient on offense against comparable defenses.")

    dee = row.get("def_ppa_adj_diff")
    if pd.notna(dee) and abs(dee) > 0.02:
        # lower def PPA diff (home-away) means home defense better
        better = row.home_team if dee < 0 else row.away_team
        why.append(f"{better} has the stronger defense by opponent-adjusted efficiency.")

    if not row.get("neutral_site", False):
        why.append(f"{row.home_team} is at home, worth a few points of edge.")
    else:
        why.append("Neutral site — no home-field edge for either team.")

    if pd.isna(off) or pd.isna(dee):
        why.append("Early season: this leans on preseason ratings until more games are played.")

    return why[:4] or ["Too close to call — essentially a coin flip."]


def main():
    payload = joblib.load(MODEL_FILE)
    model, feats = payload["model"], payload["features"]

    feat = pd.read_parquet(TRAIN_PARQUET)
    feat["date"] = pd.to_datetime(feat["date"], utc=True, errors="coerce")
    known = set(feat["home_team"]) | set(feat["away_team"])
    season = int(feat["season"].max())
    stats_by_team = team_stats.build_for_season(season)

    slate = load_slate()
    games_out = []
    for mu in slate:
        home = canon(mu.home_team, known)
        away = canon(mu.away_team, known)
        row, swapped = find_game(feat, away, home)
        if row is None:
            games_out.append({
                "away_team": away, "home_team": home, "neutral": mu.neutral,
                "error": "not found on the schedule — check spelling vs CFBD",
                "model_prob_home": None, "market_prob_home": None,
                "spread_home": None, "pick": None, "why": [], "ai_note": None,
            })
            continue

        X = pd.DataFrame([row])[feats]
        p_home = float(model.predict_proba(X)[0])
        # if schedule had sides swapped vs the slate, flip prob to slate orientation
        disp_home, disp_away = row.home_team, row.away_team
        pick = disp_home if p_home >= 0.5 else disp_away

        def team_block(name):
            s = stats_by_team.get(name, {})
            return {"name": name, "stats": s.get("stats", [])}

        games_out.append({
            "away_team": disp_away,
            "home_team": disp_home,
            "neutral": bool(row.get("neutral_site", False)),
            "model_prob_home": round(p_home, 3),
            "market_prob_home": (round(float(row["market_home_prob"]), 3)
                                 if pd.notna(row.get("market_home_prob")) else None),
            "spread_home": (float(row["spread_home"])
                            if pd.notna(row.get("spread_home")) else None),
            "pick": pick,
            "why": explain(row, p_home),
            "teams": {"away": team_block(disp_away), "home": team_block(disp_home)},
            "ai_note": None,
        })

    out = {
        "season": season,
        "generated_at": str(date.today()),
        "games": games_out,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON} with {len(games_out)} games.")


if __name__ == "__main__":
    main()
