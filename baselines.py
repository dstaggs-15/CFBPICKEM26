"""
baselines.py — the bar the model must clear.

Every predictor in this project implements the same tiny interface:

    fit(train_df)                -> self
    predict_proba(games_df)      -> np.ndarray of P(home win), one per row

That uniformity is deliberate: the Elo baseline, the market baseline, and next
week's fancy gradient-boosted model all plug into the identical harness and get
judged by the identical yardstick. Last year there was no baseline in the loop
at all, so nobody noticed the model was WORSE than one line of Elo math.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class Model:
    """Interface marker. Subclasses implement fit + predict_proba."""
    name = "base"

    def fit(self, train_df: pd.DataFrame) -> "Model":
        return self

    def predict_proba(self, games_df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class EloModel(Model):
    """
    Self-contained Elo. Walk-forward by construction: ratings only ever update
    from games already played, so predicting week N never sees week N's result.
    """
    name = "elo"

    def __init__(self, k_early=32, k_late=25, hfa=65, regress=0.5, early_week=4, base=1500.0):
        self.k_early, self.k_late = k_early, k_late
        self.hfa, self.regress, self.early_week, self.base = hfa, regress, early_week, base
        self.ratings: dict[str, float] = {}
        self.season_end: dict[int, dict[str, float]] = {}
        self._fitted_through_season = None

    @staticmethod
    def _p(elo_home, elo_away):
        return 1.0 / (10 ** (-(elo_home - elo_away) / 400) + 1)

    def _k(self, week):
        return self.k_early if week <= self.early_week else self.k_late

    def _run(self, df: pd.DataFrame, update: bool):
        """Iterate games in date order. Optionally update ratings from results."""
        probs = np.full(len(df), np.nan)
        cur_season = self._fitted_through_season
        for pos, (_, row) in enumerate(df.iterrows()):
            season = int(row["season"])
            if season != cur_season:
                if cur_season is not None:
                    self.season_end[cur_season] = dict(self.ratings)
                for t in self.ratings:
                    self.ratings[t] = (1 - self.regress) * self.ratings[t] + self.regress * self.base
                cur_season = season

            h, a = row["home_team"], row["away_team"]
            eh = self.ratings.get(h, self.base)
            ea = self.ratings.get(a, self.base)
            eh_adj = eh + (0 if row["neutral_site"] else self.hfa)
            probs[pos] = self._p(eh_adj, ea)

            if update and pd.notna(row.get("home_points")) and pd.notna(row.get("away_points")):
                k = self._k(int(row["week"]))
                home_won = row["home_points"] > row["away_points"]
                exp_home = self._p(eh_adj, ea)
                delta = k * ((1 if home_won else 0) - exp_home)
                self.ratings[h] = eh + delta
                self.ratings[a] = ea - delta
        self._fitted_through_season = cur_season
        return probs

    def fit(self, train_df: pd.DataFrame) -> "EloModel":
        df = train_df.sort_values("date").reset_index(drop=True)
        self._run(df, update=True)
        return self

    def predict_proba(self, games_df: pd.DataFrame) -> np.ndarray:
        # Predict without mutating fitted ratings (snapshot/restore).
        snap_r, snap_s, snap_season = dict(self.ratings), dict(self.season_end), self._fitted_through_season
        df = games_df.sort_values("date")
        probs = self._run(df, update=False)
        self.ratings, self.season_end, self._fitted_through_season = snap_r, snap_s, snap_season
        # restore original row order
        return pd.Series(probs, index=df.index).reindex(games_df.index).to_numpy()


class MarketModel(Model):
    """
    The market's own implied probability. If a precomputed market_home_prob
    exists, use it. Otherwise fit a logistic mapping from spread -> P(home win)
    on the training games. This is the benchmark the v1 model is forbidden to
    peek at as a feature, but must beat to justify its existence.
    """
    name = "market"

    def __init__(self):
        self.a, self.b = 0.0, 0.15

    def fit(self, train_df: pd.DataFrame) -> "MarketModel":
        d = train_df.dropna(subset=["spread_home", "home_points", "away_points"]).copy()
        if len(d) >= 50:
            spreads = d["spread_home"].to_numpy(dtype=float)
            y = (d["home_points"] > d["away_points"]).astype(int).to_numpy()
            a, b, lr = 0.0, 0.15, 0.01
            for _ in range(3000):
                z = a + b * (-spreads)  # negative spread = home favored
                q = 1 / (1 + np.exp(-z))
                a -= lr * np.mean(q - y)
                b -= lr * np.mean((q - y) * (-spreads))
            self.a, self.b = float(a), float(b)
        return self

    def predict_proba(self, games_df: pd.DataFrame) -> np.ndarray:
        if "market_home_prob" in games_df and games_df["market_home_prob"].notna().any():
            return games_df["market_home_prob"].to_numpy(dtype=float)
        spreads = games_df["spread_home"].to_numpy(dtype=float)
        return 1 / (1 + np.exp(-(self.a + self.b * (-spreads))))
