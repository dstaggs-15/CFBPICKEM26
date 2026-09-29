"""Build a factual, pregame upset watch from the existing weekly predictions.

The watch is a separate explanation, not an extra input to the win model.
Optional ESPN pick percentages are read only when their season/week matches
the prediction slate; the line identifies the underdog, not the model winner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PREDICTIONS = Path("docs/predictions.json")
TRAINING = Path("data/derived/training.parquet")
CROWD = Path("docs/input/crowd_picks.json")
OUTPUT = Path("docs/upset_watch.json")


def crowd_lookup(predictions: dict, path: Path = CROWD) -> tuple[dict, str | None]:
    if not path.exists():
        return {}, None
    snapshot = json.loads(path.read_text())
    if (snapshot.get("season"), snapshot.get("week")) != (
            predictions.get("season"), predictions.get("week")):
        print("Crowd snapshot belongs to another week; ignoring it.")
        return {}, None
    result = {}
    for g in snapshot.get("games", []):
        away, home = int(g["away_picked_pct"]), int(g["home_picked_pct"])
        if away < 0 or home < 0 or away + home != 100:
            raise ValueError(f"Bad pick percentages for {g['away']} @ {g['home']}")
        key = (g["away"], g["home"])
        if key in result:
            raise ValueError(f"Duplicate crowd matchup: {key}")
        result[key] = {g["away"]: away, g["home"]: home}
    return result, snapshot.get("source")


def historical_group(df: pd.DataFrame, current_season: int, line_abs: float,
                     dog_home: bool, matchup_edge: bool | None) -> dict:
    past = df[(df.season < current_season) &
              df.home_classification.str.lower().eq("fbs") &
              df.away_classification.str.lower().eq("fbs")].dropna(
                  subset=["home_points", "away_points", "spread_home"]).copy()
    past = past[past.spread_home.ne(0)]
    # Fixed buckets keep the rule stable as later seasons are added.
    bounds = [(0, 3), (3, 7), (7, 14), (14, float("inf"))]
    lo, hi = next((a, b) for a, b in bounds if a < line_abs <= b)
    selected = past[(past.spread_home.abs() > lo) &
                    (past.spread_home.abs() <= hi)]
    bucket_count = len(selected)
    selected = selected[selected.spread_home.gt(0).eq(dog_home)]
    details = "similar spread and underdog venue"
    if matchup_edge is not None:
        home_adv = selected.home_off_vs_away_def_ppa > selected.away_off_vs_home_def_ppa
        dog_adv = home_adv.eq(selected.spread_home.gt(0))
        narrower = selected[dog_adv.eq(matchup_edge) &
                            selected.home_off_vs_away_def_ppa.notna() &
                            selected.away_off_vs_home_def_ppa.notna()]
        if len(narrower) >= 75:
            selected = narrower
            details += ", and the same matchup-edge direction"
    dog_wins = selected.home_points.gt(selected.away_points).eq(selected.spread_home.gt(0))
    return {"count": len(selected), "dog_wins": int(dog_wins.sum()),
            "dog_win_rate": round(float(dog_wins.mean()), 3) if len(selected) else None,
            "criteria": details, "spread_bucket": f"{lo:g}–{hi:g}" if hi != float("inf") else "14+",
            "bucket_count": bucket_count}


def build(predictions: dict, features: pd.DataFrame, crowd: dict, source: str | None) -> dict:
    lookup = features.set_index("game_id")
    watches = []
    for g in predictions.get("games", []):
        line = g.get("spread_home")
        if line is None or abs(line) < 0.01 or g.get("pick") is None:
            continue
        dog_home = line > 0
        dog = g["home_team"] if dog_home else g["away_team"]
        favorite = g["away_team"] if dog_home else g["home_team"]
        if g["pick"] != dog or g.get("game_id") not in lookup.index:
            continue
        row = lookup.loc[g["game_id"]]
        dog_matchup = row["home_off_vs_away_def_ppa"] if dog_home else row["away_off_vs_home_def_ppa"]
        fav_matchup = row["away_off_vs_home_def_ppa"] if dog_home else row["home_off_vs_away_def_ppa"]
        edge = bool(dog_matchup > fav_matchup) if pd.notna(dog_matchup) and pd.notna(fav_matchup) else None
        dog_elo = row["elo_home_prob"] if dog_home else 1 - row["elo_home_prob"]
        clues = []
        if edge is not None:
            leader = dog if edge else favorite
            clues.append(f"The current offense-versus-defense PPA comparison favors {leader} "
                         f"({dog_matchup:.2f} for {dog} vs {fav_matchup:.2f} for {favorite}).")
        if pd.notna(dog_elo):
            clues.append(f"Pregame Elo favors {dog if dog_elo > .5 else favorite} "
                         f"({dog_elo:.0%} for {dog}).")
        if not g.get("neutral"):
            clues.append(f"{dog if dog_home else favorite} is at home.")
        picked = crowd.get((g["away_team"], g["home_team"]), {}).get(dog)
        dog_p = g["model_prob_home"] if dog_home else 1 - g["model_prob_home"]
        watches.append({"game_id": g["game_id"], "game_date": g.get("game_date"),
                        "underdog": dog, "favorite": favorite, "away_team": g["away_team"],
                        "home_team": g["home_team"], "spread_for_underdog": abs(line),
                        "model_prob_underdog": round(dog_p, 3),
                        "crowd_picked_pct": picked,
                        "model_vs_crowd_pp": round(dog_p * 100 - picked, 1) if picked is not None else None,
                        "clues": clues,
                        "historical": historical_group(features, int(predictions["season"]),
                                                        abs(line), dog_home, edge)})
    watches.sort(key=lambda x: (x["crowd_picked_pct"] is None,
                                x["crowd_picked_pct"] if x["crowd_picked_pct"] is not None else 101))
    return {"season": predictions["season"], "week": predictions["week"],
            "generated_at": predictions.get("generated_at"), "crowd_source": source,
            "games": watches}


def main():
    predictions = json.loads(PREDICTIONS.read_text())
    crowd, source = crowd_lookup(predictions)
    features = pd.read_parquet(TRAINING)
    payload = build(predictions, features, crowd, source)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT} with {len(payload['games'])} model-picked underdogs.")


if __name__ == "__main__":
    main()
