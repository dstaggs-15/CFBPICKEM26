"""Small, read-only current-week signal from the separate Top 25 model.

The ranking site is a résumé model, not a matchup forecast. Only compare two
teams when both have published composite scores; unranked does not mean #26.
"""

from datetime import datetime, timedelta, timezone
from math import isfinite

RANKINGS_URL = "https://dstaggs-15.github.io/cfbranking/data/rankings.json"
MAX_ADJUSTMENT = 0.02  # probability points: at most two percentage points


def load_rankings(season: int):
    import requests
    response = requests.get(RANKINGS_URL, timeout=15)
    response.raise_for_status()
    payload = response.json()
    if payload.get("season") != season:
        raise ValueError("Rankings season does not match the slate")
    published = datetime.fromisoformat(payload["last_build_utc"].replace("Z", "+00:00"))
    if published.tzinfo is None or published > datetime.now(timezone.utc):
        raise ValueError("Rankings timestamp is invalid")
    scores = {entry["team"]: float(entry["score"]) for entry in payload["top25"]}
    if not all(isfinite(score) for score in scores.values()):
        raise ValueError("Ranking scores must be finite")
    return published, scores


def adjustment(home: str, away: str, kickoff, published: datetime, scores: dict) -> float:
    if home not in scores or away not in scores:
        return 0.0
    game_time = kickoff.to_pydatetime() if hasattr(kickoff, "to_pydatetime") else kickoff
    if game_time is None or not hasattr(game_time, "tzinfo") or game_time.tzinfo is None:
        return 0.0
    if not (game_time - timedelta(days=14) <= published < game_time):
        return 0.0  # never use rankings published after kickoff
    return max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT,
                                   0.04 * (scores[home] - scores[away])))
