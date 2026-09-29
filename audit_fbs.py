"""Reproducible, market-free FBS error and reliability audit.

Run after building data/derived/training.parquet. Betting lines appear only
in the diagnostic comparison, never as model inputs.
"""

import numpy as np
import pandas as pd

from backtest import _metrics
from model import V1Model


def heldout(df, model_factory=V1Model, min_train_seasons=3):
    df = df.dropna(subset=["home_points", "away_points"]).sort_values(
        ["season", "week", "date", "game_id"])
    seasons = sorted(df.season.unique())
    outputs = []
    for season in seasons[min_train_seasons:]:
        train = df[df.season < season]
        test = df[df.season == season].copy()
        if len(test) < 50:
            continue
        test["p_model"] = model_factory().fit(train).predict_proba(test)
        outputs.append(test)
    return pd.concat(outputs, ignore_index=True)


def describe(label, d):
    if d.empty:
        return
    y = d.home_points.gt(d.away_points)
    p = d.p_model.to_numpy()
    correct = (p >= 0.5) == y.to_numpy()
    confidence = np.maximum(p, 1 - p)
    m = _metrics(y, p)
    print(f"{label:<24} n={len(d):5} accuracy={m['acc']:.3f} "
          f"Brier={m['brier']:.4f} mean_confidence={confidence.mean():.3f} "
          f"confident_80_errors={((~correct) & (confidence >= .8)).sum()}")


def main():
    df = pd.read_parquet("data/derived/training.parquet")
    games = heldout(df)
    fbs = games[games.home_classification.str.lower().eq("fbs") &
                games.away_classification.str.lower().eq("fbs")].copy()
    fbs["confidence"] = np.maximum(fbs.p_model, 1 - fbs.p_model)
    fbs["correct"] = (fbs.p_model.ge(.5) == fbs.home_points.gt(fbs.away_points))
    fbs["favorite_correct"] = (fbs.spread_home.le(0) ==
                                fbs.home_points.gt(fbs.away_points))
    describe("FBS overall", fbs)
    for title, groups in [
        ("Week", [("weeks 1–4", fbs.week.le(4)),
                  ("weeks 5–9", fbs.week.between(5, 9)),
                  ("weeks 10+", fbs.week.ge(10))]),
        ("Model confidence", [("50–60%", fbs.confidence.between(.5, .6, inclusive="left")),
                              ("60–70%", fbs.confidence.between(.6, .7, inclusive="left")),
                              ("70–80%", fbs.confidence.between(.7, .8, inclusive="left")),
                              ("80–90%", fbs.confidence.between(.8, .9, inclusive="left")),
                              ("90–100%", fbs.confidence.ge(.9))]),
        ("Venue", [("neutral", fbs.neutral_site), ("home field", ~fbs.neutral_site)]),
        ("Season", [(str(int(s)), fbs.season.eq(s)) for s in sorted(fbs.season.unique())]),
    ]:
        print(f"\n{title}:")
        for name, mask in groups:
            describe(name, fbs[mask])
    lined = fbs.dropna(subset=["spread_home"])
    print("\nSaved-line comparison on the same FBS games:")
    for name, sub in [("agree", lined[lined.p_model.ge(.5) == lined.spread_home.le(0)]),
                      ("disagree", lined[lined.p_model.ge(.5) != lined.spread_home.le(0)])]:
        print(f"{name}: n={len(sub)} model={sub.correct.mean():.3f} "
              f"favorite={sub.favorite_correct.mean():.3f}")
    # This is a retrospective diagnostic of a fixed 80% rule. It should be
    # repeated on new seasons before advertising it as a reliable policy.
    high = fbs[fbs.confidence.ge(.8)]
    print(f"\nFixed 80% confidence subset: {len(high)}/{len(fbs)} games "
          f"({len(high) / len(fbs):.1%}); accuracy={high.correct.mean():.3f}")


if __name__ == "__main__":
    main()
