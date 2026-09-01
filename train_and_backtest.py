"""
train_and_backtest.py — train the model on real football and grade it honestly.

Reads data/derived/training.parquet (from features.py), runs the walk-forward
backtest (model vs Elo vs market, plus the agree/disagree split), then trains a
final model on all played games and saves it to model.joblib for predict.py.

The model (HistGradientBoosting) handles missing values natively, so early-season
games where a team has no history yet keep a NaN strength feature instead of a
fake imputed one — "we don't know this team yet" is itself a signal the model can
use, and it leans on Elo + context there.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib

import schema
import contract
from backtest import walk_forward, format_report
from baselines import EloModel, MarketModel
from model import V1Model

TRAIN_PARQUET = "data/derived/training.parquet"
MODEL_OUT = "model.joblib"
MIN_PRIOR = 3


def main():
    df = pd.read_parquet(TRAIN_PARQUET)
    played = df.dropna(subset=["home_points", "away_points"]).copy()
    played = played.sort_values(["season", "week", "date"]).reset_index(drop=True)

    # The actual safety check, not a smaller stand-in. If travel_diff_km (or
    # anything else) is a dead constant, or a strength feature is silently
    # empty, this stops the run and names the exact problem.
    contract.validate(played, require_market=False, strength_min_coverage=0.30, stage="training")
    print(contract.report(played))

    print(f"Training on {len(played)} completed games, "
          f"seasons {int(played['season'].min())}-{int(played['season'].max())}.\n")

    factories = {
        "model": lambda: V1Model(),
        "elo": lambda: EloModel(),
        "market": lambda: MarketModel(),
    }
    per_fold, summary, disagree = walk_forward(played, factories, min_train_seasons=3)
    print(format_report(summary, disagree))

    # Train final model on everything and persist.
    final = V1Model().fit(played)
    joblib.dump({"model": final, "features": schema.MODEL_FEATURES,
                 "trained_through": int(played["season"].max())}, MODEL_OUT)
    print(f"\nSaved final model -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
