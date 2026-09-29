"""Research-only, season-forward multi-view challenger for FBS games.

Usage:
    python ensemble_challenger.py
    python ensemble_challenger.py --heldout-cache data/derived/heldout.parquet

The optional cache must contain prior-season-only V1Model forecasts keyed by
game_id (p_model). No output from this script is used by the weekly pipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import schema
from audit_fbs import heldout


MATCHUP_FEATURES = [
    "off_ppa_adj_diff", "def_ppa_adj_diff", "success_rate_adj_diff",
    "explosiveness_adj_diff", "home_off_vs_away_def_ppa",
    "away_off_vs_home_def_ppa", "rest_diff", "neutral_site",
]
INDEPENDENT_VIEWS = ["z_model", "z_elo", "z_matchup", "z_margin"]


def _clip(values):
    return np.clip(np.asarray(values, dtype=float), .001, .999)


def build_forecasts(games, independent_predictions):
    """Fit base views separately on seasons strictly before each test season."""
    games = games.dropna(subset=["home_points", "away_points"]).sort_values(
        ["season", "week", "date", "game_id"]).copy()
    games["y"] = games.home_points.gt(games.away_points).astype(int)
    games["margin"] = (games.home_points - games.away_points).clip(-50, 50)
    parts = []
    for year in sorted(games.season.unique())[3:]:
        train = games[games.season < year]
        test = games[games.season == year].copy()
        if len(test) < 50:
            continue
        margin = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                               StandardScaler(), Ridge(alpha=1000))
        matchup = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
        margin.fit(train[schema.MODEL_FEATURES], train.margin)
        matchup.fit(train[MATCHUP_FEATURES], train.y)
        test["margin_projection"] = margin.predict(test[schema.MODEL_FEATURES])
        test["p_matchup"] = matchup.predict_proba(test[MATCHUP_FEATURES])[:, 1]
        parts.append(test)
    result = pd.concat(parts, ignore_index=True).merge(
        independent_predictions[["game_id", "p_model"]], on="game_id",
        how="inner", validate="one_to_one")
    result = result[result.home_classification.str.lower().eq("fbs") &
                    result.away_classification.str.lower().eq("fbs") &
                    result.spread_home.notna() & result.spread_home.ne(0)].copy()
    result["z_market"] = logit(_clip(result.market_home_prob))
    result["z_model"] = logit(_clip(result.p_model))
    result["z_elo"] = logit(_clip(result.elo_home_prob))
    result["z_matchup"] = logit(_clip(result.p_matchup))
    result["z_margin"] = result.margin_projection / 14.
    return result


def evaluate(games):
    """Fit the combination on earlier out-of-season base forecasts only."""
    parts = []
    for year in sorted(games.season.unique()):
        train = games[games.season < year]
        test = games[games.season == year].copy()
        if len(train) < 1500 or len(test) < 100:
            continue
        for name, columns in [
            ("p_market_refit", ["z_market"]),
            ("p_independent", INDEPENDENT_VIEWS),
            ("p_hybrid", ["z_market"] + INDEPENDENT_VIEWS),
        ]:
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=.1, max_iter=1000))
            model.fit(train[columns], train.y)
            test[name] = model.predict_proba(test[columns])[:, 1]
        test["p_blend25"] = .75 * test.market_home_prob + .25 * test.p_model
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def report(group, label):
    print(f"\n{label}: {len(group)} games")
    for name in ["market_home_prob", "p_model", "p_matchup", "p_blend25",
                 "p_market_refit", "p_independent", "p_hybrid"]:
        p = _clip(group[name])
        y = group.y.to_numpy()
        print(f"  {name:<20} {int(np.sum((p >= .5) == y)):4} correct "
              f"({np.mean((p >= .5) == y):.2%})  "
              f"Brier {brier_score_loss(y, p):.5f}  log loss {log_loss(y, p):.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path,
                        default=Path("data/derived/training.parquet"))
    parser.add_argument("--heldout-cache", type=Path,
                        help="Optional cached prior-season-only V1Model predictions")
    parser.add_argument("--output", type=Path, help="Optional diagnostic Parquet")
    args = parser.parse_args()
    games = pd.read_parquet(args.training)
    forecasts = (pd.read_parquet(args.heldout_cache) if args.heldout_cache
                 else heldout(games))
    result = evaluate(build_forecasts(games, forecasts))
    report(result[result.season.between(2020, 2025)], "2020–2025")
    report(result[result.season.between(2023, 2025)], "2023–2025")
    for year, part in result.groupby("season"):
        report(part, str(year))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
