"""Optional same-game diagnostics for market-free feature experiments."""

import pandas as pd

import schema
from backtest import walk_forward, _metrics
from baselines import MarketModel
from model import V1Model

OLD_FEATURES = ["is_postseason", "rest_diff", "off_ppa_adj_diff",
                "def_ppa_adj_diff", "success_rate_adj_diff",
                "explosiveness_adj_diff", "elo_home_prob"]


def report(label, df):
    if df.empty:
        print(f"{label}: no games")
        return
    _, _, _, games = walk_forward(df, {
        "old_inputs": lambda: V1Model(features=OLD_FEATURES),
        "candidate": lambda: V1Model(features=schema.MODEL_FEATURES),
        "market": lambda: MarketModel(),
    }, min_train_seasons=3, return_games=True)
    print(f"\n{label}: {len(games)} held-out games")
    for name in ("old_inputs", "candidate", "market"):
        m = _metrics(games.home_win, games[f"p_{name}"])
        print(f"  {name:<12} n={m['n']:5d} accuracy={m['acc']:.4f} "
              f"Brier={m['brier']:.4f} logloss={m['logloss']:.4f}")
    paired = games.dropna(subset=["p_market"])
    print(f"  With saved line on identical {len(paired)} games:")
    for name in ("old_inputs", "candidate", "market"):
        m = _metrics(paired.home_win, paired[f"p_{name}"])
        print(f"    {name:<12} accuracy={m['acc']:.4f} Brier={m['brier']:.4f}")
    if "off_ppa_adj_diff" in games:
        for label, sub in [("strength present", games[games.off_ppa_adj_diff.notna()]),
                           ("strength absent", games[games.off_ppa_adj_diff.isna()])]:
            m = _metrics(sub.home_win, sub.p_candidate)
            print(f"  {label:<17} n={m['n']:5d} accuracy={m['acc']:.4f}")


def main():
    df = pd.read_parquet("data/derived/training.parquet")
    df = df.dropna(subset=["home_points", "away_points"])
    pair = (df.home_classification.str.lower().eq("fbs") &
            df.away_classification.str.lower().eq("fbs"))
    print(f"Completed games: {len(df)}; FBS-vs-FBS: {pair.sum()}; "
          f"other/unknown: {(~pair).sum()}")
    print(f"Advanced feature availability: all {df.off_ppa_adj_diff.notna().mean():.1%}; "
          f"FBS pairs {df.loc[pair, 'off_ppa_adj_diff'].notna().mean():.1%}")
    report("All classifications", df)
    report("FBS-versus-FBS only (trained and tested on FBS)", df[pair])


if __name__ == "__main__":
    main()
