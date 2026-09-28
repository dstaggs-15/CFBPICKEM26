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
cupcakes. Offensive numbers are adjusted for the defenses faced; defensive
numbers are adjusted for the offenses faced. Only information available before
each game enters the adjustment.
"""

from __future__ import annotations
from collections import deque
import numpy as np
import pandas as pd

from baselines import EloModel

BASE_PARQUET = "data/derived/games_base.parquet"
ADV_PARQUET = "data/raw/advanced_raw.parquet"
OUT_PARQUET = "data/derived/training.parquet"

ROLL_N = 8          # games of history to average over
MIN_PRIOR = 3       # need at least this many prior games before a team's stats count


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
    order = base[["game_id", "season", "week", "date", "home_team", "away_team",
                  "neutral_site", "home_points", "away_points"]].copy()

    # map each game to its two teams + who the opponent is
    home = order.rename(columns={"home_team": "team", "away_team": "opponent"})
    home["is_home"] = True
    away = order.rename(columns={"away_team": "team", "home_team": "opponent"})
    away["is_home"] = False
    tg = pd.concat([home, away], ignore_index=True)

    # attach raw efficiency for that team in that game
    metrics = ["off_ppa", "off_success", "off_explosive", "def_ppa",
               "def_success", "def_explosive"]
    a = adv[["game_id", "team", *metrics]]
    tg = tg.merge(a, on=["game_id", "team"], how="left")

    tg = tg.sort_values(["team", "season", "week", "date"]).reset_index(drop=True)

    # rolling means using only PRIOR games (shift(1) before rolling)
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
def _opponent_adjust(tg: pd.DataFrame, metric_roll: str,
                     opponent_metric_roll: str) -> pd.Series:
    """
    Adjust each team-game rolling metric by the quality of opponents faced
    SO FAR — never opponents faced later in the season or in future years.

    Compare a team's recent performance with the opposing *other-side* unit
    faced in its last eight games. Each opposing rating is frozen at the time
    of that game. Use only prior games to compute the reference league mean.
    """
    tg_sorted = tg.sort_values(["date", "game_id", "is_home"])
    orig_index = tg_sorted.index  # remember original index before reset
    tg_s = tg_sorted.reset_index(drop=True)
    opp_metric = tg_s.set_index(["game_id", "team"])[opponent_metric_roll]

    faced: dict[str, deque] = {}
    league_sum = 0.0
    league_n = 0

    adj = np.full(len(tg_s), np.nan)
    # Snapshot a whole kickoff time before updating any game at that time.
    for _, at_kickoff in tg_s.groupby("date", sort=False, dropna=False):
        reference = league_sum / league_n if league_n else None
        updates = []
        for i, row in at_kickoff.iterrows():
            history = faced.get(row["team"])
            base_val = row[metric_roll]
            if pd.notna(base_val):
                sos = np.mean(history) if history else reference
                adj[i] = base_val - (sos - reference) if sos is not None and reference is not None else base_val
            opp_val = opp_metric.get((row["game_id"], row["opponent"]))
            if (pd.notna(opp_val) and pd.notna(row["home_points"])
                    and pd.notna(row["away_points"])):
                updates.append((row["team"], float(opp_val)))
        for team, opp_val in updates:
            faced.setdefault(team, deque(maxlen=ROLL_N)).append(opp_val)
            league_sum += opp_val
            league_n += 1

    # Map back to the ORIGINAL (pre-sort) index so the caller can assign this
    # straight onto tg without any silent misalignment.
    return pd.Series(adj, index=orig_index).reindex(tg.index)


# ---------------------------------------------------------------------------
# 3. Elo probabilities across full history — TRUE pregame only
# ---------------------------------------------------------------------------
def _elo_probs(base: pd.DataFrame) -> pd.Series:
    """
    For every game, the Elo probability must reflect ONLY games that happened
    strictly before it — never the final, fully-history-informed ratings.

    The earlier version called EloModel().fit(played) (which walks the whole
    history and ends with final ratings) and then predict_proba(update=False)
    (which uses whatever ratings are currently loaded). That combination
    handed every historical game the FINAL ratings, i.e. answers that include
    results from years after that game — a serious leak.

    The fix: replay chronologically ourselves, computing each game's
    probability from the ratings as they stood at that exact moment, then
    updating. This is the only way to guarantee point-in-time correctness.
    """
    order = base.sort_values("date")
    df = order.reset_index(drop=True)
    elo = EloModel()
    probs = np.full(len(df), np.nan)
    previous_season = None
    for i, row in df.iterrows():
        season = int(row["season"])
        if previous_season is not None and season != previous_season:
            for team, rating in elo.ratings.items():
                elo.ratings[team] = (1 - elo.regress) * rating + elo.regress * elo.base
        previous_season = season
        h, a = row["home_team"], row["away_team"]
        eh = elo.ratings.get(h, elo.base)
        ea = elo.ratings.get(a, elo.base)
        eh_adj = eh + (0 if row["neutral_site"] else elo.hfa)
        probs[i] = elo._p(eh_adj, ea)

        if pd.notna(row.get("home_points")) and pd.notna(row.get("away_points")):
            k = elo._k(int(row["week"]))
            home_won = row["home_points"] > row["away_points"]
            delta = k * ((1 if home_won else 0) - probs[i])
            elo.ratings[h] = eh + delta
            elo.ratings[a] = ea - delta

    # df was built from `order` via reset_index, so df's row i corresponds to
    # order's i-th row (in original-index order). Map probs back to base's index.
    result = pd.Series(probs, index=order.index)
    return result.reindex(base.index)


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
    for own, opposing in [("off_ppa", "def_ppa"),
                           ("def_ppa", "off_ppa"),
                           ("off_success", "def_success"),
                           ("off_explosive", "def_explosive")]:
        tg[f"{own}_adj"] = _opponent_adjust(tg, f"{own}_roll", f"{opposing}_roll")

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
    df["travel_diff_km"] = 0.0  # NOT a model feature (see schema.py note) — kept
                                  # only so old data files don't break; real venue
                                  # distance is a future addition, not faked here.
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
