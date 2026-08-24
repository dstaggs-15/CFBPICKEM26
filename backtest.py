"""
backtest.py — the product.

This is the thing last year never had. It trains each model on past seasons,
predicts the next, and reports every model beside the same baselines on the
same games. The number that actually matters for a pick'em pool is at the
bottom: when your model DISAGREES with the market, does it win those games?
Agreeing with the favorite is free; disagreements are the only place an edge
can live.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

EPS = 1e-6


def _metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    pick_home = p >= 0.5
    return {
        "n": int(len(y)),
        "acc": float(np.mean(pick_home == (y == 1))),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
    }


def walk_forward(df: pd.DataFrame, model_factories: dict, min_train_seasons: int = 2):
    """
    df: canonical, contract-validated, completed games only.
    model_factories: {name: callable() -> fresh Model}. A factory (not an
        instance) so each fold trains a clean model with no leakage across folds.

    Returns (per_fold_df, summary_df, disagreement_df).
    """
    df = df.sort_values(["season", "week", "date"]).reset_index(drop=True)
    df["home_win"] = (df["home_points"] > df["away_points"]).astype(int)
    seasons = sorted(df["season"].unique())

    rows = []
    keep = []  # per-game predictions for the disagreement analysis
    for i, test_season in enumerate(seasons):
        if i < min_train_seasons:
            continue
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season]
        if len(test) < 50:
            continue

        preds = {}
        for name, factory in model_factories.items():
            model = factory().fit(train)
            preds[name] = model.predict_proba(test)
            m = _metrics(test["home_win"], preds[name])
            rows.append({"test_season": test_season, "model": name, **m})

        fold = test[["season", "week", "home_team", "away_team", "home_win"]].copy()
        for name, p in preds.items():
            fold[f"p_{name}"] = p
        keep.append(fold)

    per_fold = pd.DataFrame(rows)
    summary = (
        per_fold.groupby("model")[["acc", "brier", "logloss"]].mean()
        .join(per_fold.groupby("model")["n"].sum())
        .sort_values("acc", ascending=False)
    )
    games = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()

    disagreement = _disagreement_analysis(games)
    return per_fold, summary, disagreement


def _disagreement_analysis(games: pd.DataFrame, model="model", market="market"):
    """
    The money question. Split games by whether the model and the market pick the
    same winner. On agreements, the model can't add value (it's just echoing the
    market). All edge — good or bad — lives in the disagreements.
    """
    if games.empty or f"p_{model}" not in games or f"p_{market}" not in games:
        return pd.DataFrame()

    g = games.copy()
    g["model_home"] = g[f"p_{model}"] >= 0.5
    g["market_home"] = g[f"p_{market}"] >= 0.5
    g["agree"] = g["model_home"] == g["market_home"]
    g["model_correct"] = g["model_home"] == (g["home_win"] == 1)
    g["market_correct"] = g["market_home"] == (g["home_win"] == 1)

    out = []
    for label, sub in [("AGREE", g[g["agree"]]), ("DISAGREE", g[~g["agree"]]), ("ALL", g)]:
        if len(sub):
            out.append({
                "bucket": label,
                "games": len(sub),
                "model_acc": sub["model_correct"].mean(),
                "market_acc": sub["market_correct"].mean(),
                "model_minus_market": sub["model_correct"].mean() - sub["market_correct"].mean(),
            })
    return pd.DataFrame(out)


def format_report(summary: pd.DataFrame, disagreement: pd.DataFrame) -> str:
    lines = ["", "=" * 60, "WALK-FORWARD BACKTEST", "=" * 60, "", "Overall (averaged across test seasons):", ""]
    lines.append(f"{'model':<12}{'acc':>8}{'brier':>9}{'logloss':>9}{'games':>8}")
    for name, r in summary.iterrows():
        lines.append(f"{name:<12}{r['acc']:>8.3f}{r['brier']:>9.4f}{r['logloss']:>9.4f}{int(r['n']):>8}")

    if not disagreement.empty:
        lines += ["", "-" * 60, "Model vs Market — where the edge actually is:", ""]
        lines.append(f"{'bucket':<10}{'games':>7}{'model':>9}{'market':>9}{'edge':>9}")
        for _, r in disagreement.iterrows():
            lines.append(
                f"{r['bucket']:<10}{int(r['games']):>7}{r['model_acc']:>9.3f}"
                f"{r['market_acc']:>9.3f}{r['model_minus_market']:>+9.3f}"
            )
        lines += ["", "Read the DISAGREE row. Positive 'edge' there = a real, tradeable signal.",
                  "Negative = the model is confidently wrong exactly when it matters."]
    return "\n".join(lines)
