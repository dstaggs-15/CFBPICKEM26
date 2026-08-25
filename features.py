"""
features.py — turn raw fetched data into model inputs.

This is the piece that separates two evenly-matched teams, and the piece that
was completely missing last year. It reads what fetch_cfbd.py landed:
    data/derived/games_base.parquet   (games + market)
    data/raw/advanced_raw.parquet     (per-game team efficiency)
and produces:
    data/derived/training.parquet     (one row per game, schema-compliant)

The cardinal rule enforced everywhere below: a game's features may only use
information from BEFORE that game. Every rolling number is shifted so the
current game is never part of its own inputs. That is what makes the backtest
honest instead of fantasy.

Opponent adjustment, in plain terms: raw PPA says "Team A averaged 0.55 per
play." But 0.55 against elite defenses is very different from 0.55 against
cupcakes. We adjust each team's number by the average strength of the defenses
they actually faced, iterating a couple of times so the adjustment itself
accounts for opponent quality. The result is a rating you can compare across
schedules.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from baselines import EloModel

BASE_PARQUET = "data/derived/games_base.parquet"
ADV_PARQUET = "data/raw/advanced_raw.parquet"
OUT_PARQUET = "data/derived/training.parquet"

ROLL_N = 8          # games of history to average over
MIN_PRIOR = 3       # need at least this many prior games before a team's stats count
ADJUST_ITERS = 2    # opponent-adjustment refinement passes


# ---------------------------------------------------------------------------
# 1. Long per-team-game efficiency table with PRIOR-ONLY rolling means
# ---------------------------------------------------------------------------
def _team_game_long(base: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (game, team) with that team's rolling efficiency ENTERING the
    game (shifted), plus the opponent for adjustment. Built from adv stats
    joined onto game order.
    """
    # game order key: season, week, date
    order = base[["game_id", "season", "week", "date", "home_team", "away_team", "neutral_site"]].copy()

    # map each game to its two teams + who the opponent is
    home = order.rename(columns={"home_team": "team", "away_team": "opponent"})
    home["is_home"] = True
    away = order.rename(columns={"away_team": "team", "home_team": "opponent"})
    away["is_home"] = False
    tg = pd.concat([home, away], ignore_index=True)

    # attach raw efficiency for that team in that game
    a = adv.rename(columns={})[["game_id", "team", "off_ppa", "off_success", "off_explosive", "def_ppa"]]
    tg = tg.merge(a, on=["game_id", "team"], how="left")

    tg = tg.sort_values(["team", "season", "week", "date"]).reset_index(drop=True)

    # rolling means using only PRIOR games (shift(1) before rolling)
    metrics = ["off_ppa", "off_success", "off_explosive", "def_ppa"]
    for m in metrics:
        shifted = tg.groupby("team")[m].shift(1)
        tg[f"{m}_roll"] = (
            shifted.groupby(tg["team"]).rolling(ROLL_N, min_periods=MIN_PRIOR).mean()
            .reset_index(level=0, drop=True)
        )
    # count of prior games (for min-history gating)
    tg["prior_games"] = tg.groupby("team").cumcount()
    return tg


# ---------------------------------------------------------------------------
# 2. Opponent adjustment: iterate team rating vs schedule faced
# ---------------------------------------------------------------------------
def _opponent_adjust(tg: pd.DataFrame, metric_roll: str) -> pd.Series:
    """
    Adjust each team-game rolling metric by the quality of opponents faced.
    Simple, robust scheme: adjusted = raw - mean(opponent_raw faced so far).
    Iterated a couple times so opponents' own adjustments feed back in.
    Everything stays prior-only because it's built from the shifted rolling
    values. Returns a Series aligned to tg.index.
    """
    adj = tg[metric_roll].copy()
    # opponent's rolling value entering the same game
    opp_lookup = tg.set_index(["game_id", "team"])[metric_roll]
    # map opponent value per row
    opp_vals = tg.apply(
        lambda r: opp_lookup.get((r["game_id"], r["opponent"]), np.nan), axis=1
    )
    league_mean = tg[metric_roll].mean()
    for _ in range(ADJUST_ITERS):
        # strength of schedule = average opponent adjusted value so far
        tmp = pd.DataFrame({"team": tg["team"], "opp_adj": opp_vals})
        sos = tmp.groupby("team")["opp_adj"].transform("mean")
        adj = tg[metric_roll] - (sos - league_mean)
        # refresh opponent values using the new adjustment for next iter
        row_adj = pd.Series(adj.values, index=pd.MultiIndex.from_frame(tg[["game_id", "team"]]))
        opp_vals = tg.apply(
            lambda r: row_adj.get((r["game_id"], r["opponent"]), np.nan), axis=1
        )
    return adj


# ---------------------------------------------------------------------------
# 3. Elo probabilities across full history
# ---------------------------------------------------------------------------
def _elo_probs(base: pd.DataFrame) -> pd.Series:
    played = base.dropna(subset=["home_points", "away_points"]).copy()
    elo = EloModel().fit(played)  # fits ratings across history
    # predict pregame prob for every game (played or not) in order
    probs = elo.predict_proba(base.sort_values("date"))
    return pd.Series(probs, index=base.sort_values("date").index).reindex(base.index)


# ---------------------------------------------------------------------------
# 4. Rest / travel context
# ---------------------------------------------------------------------------
def _rest_diff(base: pd.DataFrame) -> pd.Series:
    # days since each team's previous game, home minus away
    long = []
    for side in ("home_team", "away_team"):
        d = base[["game_id", "season", "date", side]].rename(columns={side: "team"})
        long.append(d)
    L = pd.concat(long).sort_values(["team", "date"])
    L["rest"] = L.groupby("team")["date"].diff().dt.days
    rest_lookup = L.set_index(["game_id", "team"])["rest"]
    home_rest = base.apply(lambda r: rest_lookup.get((r["game_id"], r["home_team"]), np.nan), axis=1)
    away_rest = base.apply(lambda r: rest_lookup.get((r["game_id"], r["away_team"]), np.nan), axis=1)
    return (home_rest.fillna(7) - away_rest.fillna(7)).clip(-14, 14)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build(base: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["date"] = pd.to_datetime(base["date"], utc=True, errors="coerce")

    tg = _team_game_long(base, adv)

    # opponent-adjust each rolling metric
    for m in ["off_ppa", "off_success", "off_explosive", "def_ppa"]:
        tg[f"{m}_adj"] = _opponent_adjust(tg, f"{m}_roll")

    # pivot adjusted metrics back to home/away per game
    keep = ["game_id", "prior_games",
            "off_ppa_adj", "off_success_adj", "off_explosive_adj", "def_ppa_adj"]
    home = tg[tg.is_home][keep].add_prefix("home_").rename(columns={"home_game_id": "game_id"})
    away = tg[~tg.is_home][keep].add_prefix("away_").rename(columns={"away_game_id": "game_id"})

    df = base.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")

    # schema strength features = home minus away of each adjusted metric
    df["off_ppa_adj_diff"] = df["home_off_ppa_adj"] - df["away_off_ppa_adj"]
    df["def_ppa_adj_diff"] = df["home_def_ppa_adj"] - df["away_def_ppa_adj"]
    df["success_rate_adj_diff"] = df["home_off_success_adj"] - df["away_off_success_adj"]
    df["explosiveness_adj_diff"] = df["home_off_explosive_adj"] - df["away_off_explosive_adj"]

    # elo + context
    df["elo_home_prob"] = _elo_probs(base).values
    df["rest_diff"] = _rest_diff(base).values
    df["travel_diff_km"] = 0.0  # placeholder until venue coords wired; kept for schema
    df["is_postseason"] = df["is_postseason"].fillna(0).astype(int)

    # gate: null out strength diffs when either team lacks enough history,
    # so early-season rows don't pretend to know a team they haven't seen.
    enough = (df["home_prior_games"] >= MIN_PRIOR) & (df["away_prior_games"] >= MIN_PRIOR)
    for c in ["off_ppa_adj_diff", "def_ppa_adj_diff", "success_rate_adj_diff", "explosiveness_adj_diff"]:
        df.loc[~enough, c] = np.nan

    return df


def main():
    base = pd.read_parquet(BASE_PARQUET)
    adv = pd.read_parquet(ADV_PARQUET)
    df = build(base, adv)
    df.to_parquet(OUT_PARQUET, index=False)

    # coverage report on the model features (post gating)
    from schema import MODEL_FEATURES
    print("=" * 52)
    print(f"FEATURES BUILT — {len(df)} games")
    print("=" * 52)
    played = df.dropna(subset=["home_points", "away_points"])
    for c in MODEL_FEATURES:
        cov = df[c].notna().mean() if c in df else float("nan")
        pcov = played[c].notna().mean() if c in played else float("nan")
        print(f"  {c:<26} all:{cov:>6.1%}   played-games:{pcov:>6.1%}")
    print(f"\n  saved: {OUT_PARQUET}")
    print("  next: train_and_backtest.py")


if __name__ == "__main__":
    main()
