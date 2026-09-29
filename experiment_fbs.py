"""Compare proposed pregame features on identical held-out FBS games."""

import pandas as pd

import schema
from backtest import _metrics, walk_forward
from model import V1Model


def main():
    df = pd.read_parquet("data/derived/training.parquet").dropna(
        subset=["home_points", "away_points"])
    names = {
        "current": lambda: V1Model(),
        "pass_rush": lambda: V1Model(
            features=schema.MODEL_FEATURES + schema.EXPERIMENTAL_MATCHUP_FEATURES),
        "pass_rush_sigmoid": lambda: V1Model(
            features=schema.MODEL_FEATURES + schema.EXPERIMENTAL_MATCHUP_FEATURES,
            calibration="sigmoid"),
        "current_sigmoid": lambda: V1Model(calibration="sigmoid"),
    }
    _, _, _, games = walk_forward(df, names, min_train_seasons=3, return_games=True)
    fbs = games[games.home_classification.str.lower().eq("fbs") &
                games.away_classification.str.lower().eq("fbs")]
    for label, part in [("FBS", fbs),
                        ("early (weeks 1–4)", fbs[fbs.week.le(4)]),
                        ("neutral", fbs[fbs.neutral_site])]:
        print(f"\n{label} ({len(part)} games)")
        for name in names:
            m = _metrics(part.home_win, part[f"p_{name}"])
            print(f"  {name:<20} accuracy={m['acc']:.4f} "
                  f"Brier={m['brier']:.4f} logloss={m['logloss']:.4f}")
    for season, part in fbs.groupby("season"):
        print(f"\n{season} ({len(part)} games)")
        for name in names:
            m = _metrics(part.home_win, part[f"p_{name}"])
            print(f"  {name:<20} accuracy={m['acc']:.4f} Brier={m['brier']:.4f}")
    print("\nPassing/rushing feature coverage on FBS pairs:")
    for name in schema.EXPERIMENTAL_MATCHUP_FEATURES:
        print(f"  {name}: {df.loc[df.home_classification.str.lower().eq('fbs') & df.away_classification.str.lower().eq('fbs'), name].notna().mean():.1%}")


if __name__ == "__main__":
    main()
