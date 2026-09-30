"""Identity-free historical matchup examples; descriptive, not a pick override."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROFILE_FEATURES = ["home_off_ppa_adj", "away_def_ppa_adj", "away_off_ppa_adj",
                    "home_def_ppa_adj", "home_off_success_adj", "away_off_success_adj",
                    "home_off_explosive_adj", "away_off_explosive_adj"]


class ProfileAnalogs:
    def __init__(self, history, neighbors=75, features=None):
        self.history = history.copy()
        self.neighbors = neighbors
        self.features = list(features if features is not None else PROFILE_FEATURES)

    def describe(self, row):
        if any(pd.isna(row.get(c)) for c in self.features):
            return {"available": False, "reason": "Incomplete pregame efficiency profile."}
        h = self.history
        # Earlier seasons only: no current-season result or same-game outcome.
        h = h[(h.season < row.season) & (pd.to_datetime(h.date, utc=True) < row.date)
              & h.home_points.notna() & h.away_points.notna()
              & h.neutral_site.eq(row.neutral_site)].copy()
        for c in ("home_classification", "away_classification"):
            if c in h:
                h = h[h[c].str.lower().eq("fbs")]
        h = h.dropna(subset=self.features).drop_duplicates("game_id")
        if len(h) < self.neighbors:
            return {"available": False, "reason": "Too few complete earlier-season FBS profiles."}
        scaler = StandardScaler().fit(h[self.features])
        a = scaler.transform(h[self.features])
        b = scaler.transform(pd.DataFrame([row])[self.features])[0]
        distances = np.sqrt(np.mean((a - b) ** 2, axis=1))
        indices = np.argsort(distances, kind="stable")[:self.neighbors]
        selected = h.iloc[indices]
        wins = int(selected.home_points.gt(selected.away_points).sum())
        # Only the declared pregame profile inputs determine distance.
        examples = []
        for pos in indices[:3]:
            g = h.iloc[pos]
            examples.append({"season": int(g.season), "game_id": str(g.game_id),
                             "home_team": g.home_team, "away_team": g.away_team,
                             "home_points": int(g.home_points), "away_points": int(g.away_points),
                             "distance": round(float(distances[pos]), 3)})
        return {"available": True, "features": self.features, "n": len(selected), "home_wins": wins,
                "home_win_rate": round(wins / len(selected), 3),
                "median_distance": round(float(np.median(distances[indices])), 3),
                "examples": examples,
                "note": "Descriptive historical sample; this rate is not the calibrated forecast."}
