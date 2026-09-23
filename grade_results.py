"""Grade archived weekly picks against final scores from the existing CFBD fetch.

Run after fetch_cfbd.py and before predict.py. Incomplete weeks remain unchanged.
"""

import json
from pathlib import Path

import pandas as pd


BASE = Path("data/derived/games_base.parquet")
ARCHIVES = Path("historicals/predictions")
RESULTS = Path("docs/results.json")


def grade(picks, games):
    """Return a verified weekly record, or None until every game is final."""
    season = picks["season"]
    week = picks.get("week")
    slate = picks.get("games", [])
    if not slate:
        return None
    wins = losses = 0
    matched_weeks = set()
    used_ids = set()
    details = []
    for pick in slate:
        if not pick.get("pick") or pick.get("error"):
            return None
        matches = games[games.season == season]
        if pick.get("game_id"):
            matches = matches[matches.game_id.astype(str) == str(pick["game_id"])]
        else:
            matches = matches[
                ((matches.home_team == pick["home_team"]) & (matches.away_team == pick["away_team"])) |
                ((matches.home_team == pick["away_team"]) & (matches.away_team == pick["home_team"]))
            ]
        if week is not None:
            matches = matches[matches.week == week]
        if len(matches) != 1:
            return None
        game = matches.iloc[0]
        gid = str(game.game_id)
        if gid in used_ids or pd.isna(game.home_points) or pd.isna(game.away_points):
            return None
        if game.home_points == game.away_points:
            return None
        used_ids.add(gid)
        matched_weeks.add(int(game.week))
        winner = game.home_team if game.home_points > game.away_points else game.away_team
        if pick["pick"] not in (game.home_team, game.away_team):
            return None
        wins += winner == pick["pick"]
        losses += winner != pick["pick"]
        spread = pick.get("spread_home")
        favorite = (game.home_team if spread < 0 else game.away_team) if spread is not None and spread != 0 else None
        probability = pick.get("model_prob_home")
        game_date = pd.to_datetime(game.get("date"), utc=True, errors="coerce")
        details.append({
            "game_id": gid,
            "game_date": game_date.date().isoformat() if pd.notna(game_date) else pick.get("game_date"),
            "published_at": picks.get("generated_at"),
            "away_team": str(game.away_team),
            "home_team": str(game.home_team),
            "away_points": int(game.away_points),
            "home_points": int(game.home_points),
            "pick": pick["pick"],
            "model_confidence": round(max(probability, 1 - probability), 3) if probability is not None else None,
            "market_favorite": favorite,
            "winner": str(winner),
        })
    if len(matched_weeks) != 1:
        return None
    return {"week": matched_weeks.pop(), "wins": int(wins), "losses": int(losses), "source": "final scores", "picks": details}


def main():
    if not BASE.exists():
        raise SystemExit(f"Missing {BASE}; run fetch_cfbd.py first")
    games = pd.read_parquet(BASE)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"season": int(games.season.max()), "weeks": []}
    by_week = {row["week"]: row for row in results["weeks"]}
    for archive in sorted(ARCHIVES.glob("*.json")):
        picks = json.loads(archive.read_text())
        if picks["season"] != results["season"]:
            continue
        scored = grade(picks, games)
        if scored:
            by_week[scored["week"]] = scored
            print(f"Week {scored['week']}: {scored['wins']}-{scored['losses']} from final scores")
        else:
            print(f"Waiting for complete results: {archive}")
    results["weeks"] = [by_week[w] for w in sorted(by_week)]
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
