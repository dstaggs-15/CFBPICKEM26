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

The "why" describes observed pregame evidence and historical profile examples.
It does not attribute a boosted-tree decision to an individual feature.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import requests

import schema
from weekly_input import load_slate
import team_stats
from ranking_signal import adjustment, load_rankings
from profile_analogs import ProfileAnalogs

TRAIN_PARQUET = "data/derived/joint_training.parquet"
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
    if unplayed.empty:
        return None, False
    row = unplayed.sort_values("date").iloc[0]
    swapped = not (row.home_team == home and row.away_team == away)
    return row, swapped


def explain(row: pd.Series, p_home: float) -> list[str]:
    """Pregame evidence, not a claim of causal feature attribution."""
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
        if better == fav:
            why.append(f"{better} has higher opponent-adjusted offensive PPA in its recent pregame profile.")

    dee = row.get("def_ppa_adj_diff")
    if pd.notna(dee) and abs(dee) > 0.02:
        # lower def PPA diff (home-away) means home defense better
        better = row.home_team if dee < 0 else row.away_team
        if better == fav:
            why.append(f"{better} has the stronger defense by opponent-adjusted efficiency.")

    if not row.get("neutral_site", False):
        why.append(f"{row.home_team} is at home; venue is an input to the forecast.")
    else:
        why.append("Neutral site — no home-field edge for either team.")

    if pd.isna(off) or pd.isna(dee):
        why.append("Efficiency inputs are incomplete; the forecast also uses pregame Elo and game context.")

    return why[:4] or ["Too close to call — essentially a coin flip."]


def main():
    payload = joblib.load(MODEL_FILE)
    model, feats = payload["model"], payload["features"]

    feat = pd.read_parquet(TRAIN_PARQUET)
    feat["date"] = pd.to_datetime(feat["date"], utc=True, errors="coerce")
    known = set(feat["home_team"]) | set(feat["away_team"])
    season = int(feat["season"].max())
    stats_by_team = team_stats.build_for_season(season)
    joint = payload.get('model_version') == 'joint-stats-elo-v3'
    analogs = ProfileAnalogs(feat, features=(schema.JOINT_PROFILE_FEATURES + ['elo_home_prob']) if joint else None)
    try:
        rankings_published, ranking_scores = load_rankings(season)
        print(f"Using ranking snapshot published {rankings_published.isoformat()}")
    except (ValueError, KeyError, requests.RequestException) as exc:
        rankings_published, ranking_scores = None, {}
        print(f"Rankings signal unavailable; no adjustments applied: {exc}")

    slate = load_slate()
    games_out = []
    slate_weeks = set()
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

        if pd.isna(row["date"]) or row["date"] <= pd.Timestamp.now(tz="UTC"):
            raise ValueError("Refusing to publish a new forecast after kickoff")
        slate_weeks.add(int(row["week"]))

        X = pd.DataFrame([row])[feats]
        base_p_home = float(model.predict_proba(X)[0])
        rank_delta = (adjustment(row.home_team, row.away_team, row["date"],
                                 rankings_published, ranking_scores)
                      if rankings_published else 0.0)
        p_home = min(1.0, max(0.0, base_p_home + rank_delta))
        # if schedule had sides swapped vs the slate, flip prob to slate orientation
        disp_home, disp_away = row.home_team, row.away_team
        pick = disp_home if p_home >= 0.5 else disp_away

        def team_block(name):
            s = stats_by_team.get(name, {})
            return {"name": name, "stats": s.get("stats", [])}

        if joint:
            cutoff = row['date'].normalize() - pd.Timedelta(days=row['date'].weekday())
            count = int(row.get('joint_min_games', 0))
            reasons = [
                f"One fitted model combines this season's opponent-adjusted matchup stats with pregame Elo. "
                f"The statistical profile uses completed FBS games before {cutoff.date()}, with at least {count} games per team."
            ]
            if count < 2:
                reasons.append('The current-season statistical profile has too few FBS games; those inputs are marked missing, so Elo and context have more influence.')
        else:
            cutoff = None
            reasons = explain(row, p_home)
        model_terms = {}
        if hasattr(model, "contributions"):
            terms = model.contributions(X).iloc[0]
            model_terms = {str(k): round(float(v), 5) for k, v in terms.items()}
            labels = {
                "joint_off_ppa_edge": "the opponent-adjusted PPA matchup",
                "joint_off_success_edge": "the opponent-adjusted success-rate matchup",
                "joint_off_explosive_edge": "the opponent-adjusted explosiveness matchup",
                "joint_points_edge": "the opponent-adjusted scoring matchup",
                "elo_home_prob": "pregame Elo, with annual regression toward average",
                "off_ppa_adj_diff": "the opponent-adjusted offensive efficiency gap",
                "def_ppa_adj_diff": "the opponent-adjusted defensive efficiency gap",
                "success_rate_adj_diff": "the opponent-adjusted success-rate gap",
                "explosiveness_adj_diff": "the opponent-adjusted explosiveness gap",
                "home_off_vs_away_def_ppa": "the home offense / away defense matchup",
                "away_off_vs_home_def_ppa": "the away offense / home defense matchup",
                "rest_diff": "the rest-day difference", "neutral_site": "venue context",
                "is_postseason": "postseason context",
            }
            support = terms * (1 if base_p_home >= .5 else -1)
            strongest = support[support > 0].sort_values(ascending=False).head(3).index
            drivers = [labels.get(k, "the availability of pregame efficiency data") for k in strongest]
            if drivers:
                reasons.insert(0, "The fitted statistical model's strongest supporting inputs are " + "; ".join(drivers) + ".")
            opposing = support[support < 0].sort_values().head(2).index
            if joint and len(opposing):
                reasons.append("Inputs pulling toward " + (disp_away if base_p_home >= .5 else disp_home) + ": " + "; ".join(labels.get(k, 'pregame data availability') for k in opposing) + ".")
        historical_profile = analogs.describe(row)
        if historical_profile["available"]:
            n = historical_profile["n"]
            wins = historical_profile["home_wins"]
            chosen_wins = wins if p_home >= .5 else n - wins
            reasons.append(
                f"In {n} similar earlier-season FBS {'stats-and-Elo ' if joint else ''}matchup profiles, the side matching "
                f"{pick}'s role won {chosen_wins}/{n}. This is historical context, "
                "not an additional forecast probability.")
        if rank_delta:
            ranked_side = disp_home if rank_delta > 0 else disp_away
            reasons.insert(0, f"The separate computer rankings favor {ranked_side}; they shift the home win estimate by {rank_delta * 100:+.2f} percentage points.")

        games_out.append({
            "game_id": str(row["game_id"]),
            "game_date": row["date"].date().isoformat() if pd.notna(row.get("date")) else None,
            "away_team": disp_away,
            "home_team": disp_home,
            "neutral": bool(row.get("neutral_site", False)),
            "model_prob_home": round(p_home, 3),
            "base_model_prob_home": round(base_p_home, 3),
            "ranking_adjustment_home": round(rank_delta, 4),
            "market_prob_home": (round(float(row["market_home_prob"]), 3)
                                 if pd.notna(row.get("market_home_prob")) else None),
            "spread_home": (float(row["spread_home"])
                            if pd.notna(row.get("spread_home")) else None),
            "pick": pick,
            "why": reasons,
            "historical_profile": historical_profile,
            "model_log_odds_terms": model_terms,
            "model_log_odds_intercept": (round(float(model.pipeline.steps[-1][1].intercept_[0]), 8) if hasattr(model, 'pipeline') else None),
            "model_input_values": {c: (round(float(row[c]), 6) if pd.notna(row[c]) else None) for c in feats},
            "profile_snapshot_utc": cutoff.isoformat() if cutoff is not None else None,
            "teams": {"away": team_block(disp_away), "home": team_block(disp_home)},
            "ai_note": None,
        })

    out = {
        "season": season,
        "week": next(iter(slate_weeks)) if len(slate_weeks) == 1 else None,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_version": payload.get("model_version", "boosted-v1"),
        "ranking_snapshot_utc": (rankings_published.isoformat()
                                 if rankings_published else None),
        "games": games_out,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    # Save the first complete set of picks for later grading. Re-running a week
    # cannot silently replace the picks that were already published.
    if out["week"] is not None and len(games_out) == len(slate) and all(g.get("game_id") for g in games_out):
        archive = Path(f"historicals/predictions/{season}-week-{out['week']}.json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_text(json.dumps(out, indent=2) + "\n")
            print(f"Archived first picks for week {out['week']} at {archive}")
        version_archive = Path(f"historicals/model_versions/{out['model_version']}/{season}-week-{out['week']}.json")
        version_archive.parent.mkdir(parents=True, exist_ok=True)
        if not version_archive.exists():
            version_archive.write_text(json.dumps(out, indent=2) + "\n")
            print(f"Archived first picks for model version at {version_archive}")
    else:
        print("Slate has missing games or mixed weeks; no grading archive created.")
    print(f"Wrote {OUT_JSON} with {len(games_out)} games.")


if __name__ == "__main__":
    main()
