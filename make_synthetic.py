"""
make_synthetic.py — fake but structurally-honest CFB data.

Lets us prove the contract + baselines + harness end-to-end with no API key.
Teams have a hidden 'true strength'; scores, spreads, and stats all flow from
it with noise, so a good model CAN learn something and the market baseline is
genuinely informative — just like reality.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def make(seasons=range(2016, 2024), n_teams=48, games_per_team=12, seed=7,
         break_column: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"Team_{i:02d}" for i in range(n_teams)]
    rows = []
    gid = 0
    # persistent latent strength with year-to-year carryover
    strength = {t: rng.normal(0, 1) for t in teams}

    for season in seasons:
        for t in teams:
            strength[t] = 0.7 * strength[t] + 0.3 * rng.normal(0, 1)
        week = 1
        matchups_this_week = 0
        order = teams.copy()
        for _ in range(games_per_team):
            rng.shuffle(order)
            for i in range(0, n_teams - 1, 2):
                h, a = order[i], order[i + 1]
                neutral = rng.random() < 0.06
                sh, sa = strength[h], strength[a]
                hfa = 0.0 if neutral else 0.45
                margin_mean = (sh + hfa - sa) * 10.0
                margin = rng.normal(margin_mean, 14.0)
                total = rng.normal(52, 8)
                hp = max(0, round((total + margin) / 2))
                ap = max(0, round((total - margin) / 2))

                # market spread ~ true margin with noise (market is smart but imperfect)
                spread_home = -round((margin_mean + rng.normal(0, 2.5)) * 2) / 2  # neg = home fav
                mprob = 1 / (1 + np.exp(-(-spread_home) * 0.14))

                rows.append({
                    "game_id": f"g{gid}", "season": season, "week": week,
                    "date": pd.Timestamp("2016-08-25", tz="UTC")
                            + pd.to_timedelta((season - 2016) * 365 + week * 7, unit="D")
                            + pd.to_timedelta(int(rng.integers(0, 3)), unit="D"),  # realistic day-of-week spread
                    "home_team": h, "away_team": a, "neutral_site": neutral,
                    "home_points": hp, "away_points": ap,
                    "is_postseason": int(neutral and rng.random() < 0.5),
                    "rest_diff": int(rng.integers(-3, 4)),
                    "travel_diff_km": float(round(rng.uniform(0, 2500), 1)),
                    # opponent-adjusted strength proxies (home minus away), noisy views of truth
                    "off_ppa_adj_diff": (sh - sa) * 0.3 + rng.normal(0, 0.1),
                    "def_ppa_adj_diff": (sh - sa) * 0.2 + rng.normal(0, 0.1),
                    "success_rate_adj_diff": (sh - sa) * 0.05 + rng.normal(0, 0.02),
                    "explosiveness_adj_diff": (sh - sa) * 0.04 + rng.normal(0, 0.03),
                    "elo_home_prob": np.nan,  # filled by EloModel in real pipeline; not needed for synth model test
                    "spread_home": spread_home,
                    "over_under": round(total, 1),
                    "market_home_prob": float(mprob),
                })
                gid += 1
                matchups_this_week += 1
            week += 1

    df = pd.DataFrame(rows)
    # elo_home_prob is a strength feature; approximate it here so the contract passes.
    df["elo_home_prob"] = df["market_home_prob"] * 0.5 + 0.25  # placeholder, non-constant

    if break_column:
        # Simulate last year's exact failure: a feature join silently returns all-null.
        df[break_column] = np.nan
    return df

if __name__ == "__main__":
    d = make()
    print(d.head())
    print("shape:", d.shape)
