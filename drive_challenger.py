"""Research-only prior-game drive/field-position challenger.

With CFBD_API_KEY set, run:
  python drive_challenger.py --fetch --seasons 2014-2026

Runs on completed historical games. Nothing here modifies the live model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_fbs import heldout
from fetch_cfbd import _get, _get_field, _season_list
from model import V1Model
import schema


DRIVE_COLS = ["drive_off_ppd", "drive_def_ppd", "drive_off_start", "drive_def_start"]
NEW_FEATURES = [
    "drive_ppd_matchup_home", "drive_ppd_matchup_away",
    "drive_field_position_diff",
]


def parse_drives(raw):
    rows = []
    for d in raw:
        start = pd.to_numeric(_get_field(d, "startYardsToGoal"), errors="coerce")
        period = pd.to_numeric(_get_field(d, "startPeriod"), errors="coerce")
        if pd.isna(start) or not 1 <= start <= 99 or pd.isna(period) or period > 4:
            continue
        rows.append({
            "drive_id": str(_get_field(d, "id")),
            "game_id": str(_get_field(d, "gameId")),
            "offense": _get_field(d, "offense"),
            "defense": _get_field(d, "defense"),
            "start_yards_to_goal": float(start),
            "points": max(0, min(8, float(_get_field(d, "endOffenseScore", default=0)) -
                                  float(_get_field(d, "startOffenseScore", default=0)))),
        })
    return pd.DataFrame(rows, columns=["drive_id", "game_id", "offense", "defense",
                                       "start_yards_to_goal", "points"]).drop_duplicates("drive_id")


def fetch_drives(seasons, output):
    parts = []
    for year in seasons:
        raw = _get("/drives", year=year, seasonType="both")
        part = parse_drives(raw)
        print(f"{year}: {len(part)} regulation drives", flush=True)
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def drive_features(games, drives):
    """Use only *completed earlier games* to describe each team's next game."""
    d = drives.copy()
    d["points"] = pd.to_numeric(d.points, errors="coerce")
    d["start_yards_to_goal"] = pd.to_numeric(d.start_yards_to_goal, errors="coerce")
    d = d.dropna(subset=["game_id", "offense", "defense", "points",
                         "start_yards_to_goal"])
    off = d.groupby(["game_id", "offense"], as_index=False).agg(
        drive_off_ppd=("points", "mean"),
        drive_off_start=("start_yards_to_goal", "mean"),
        off_drives=("points", "size"))
    deff = d.groupby(["game_id", "defense"], as_index=False).agg(
        drive_def_ppd=("points", "mean"),
        drive_def_start=("start_yards_to_goal", "mean"),
        def_drives=("points", "size"))
    per_game = off.rename(columns={"offense": "team"}).merge(
        deff.rename(columns={"defense": "team"}), on=["game_id", "team"], how="outer")
    # Incomplete CFBD drive pages cannot silently become season form.
    for side in ["off", "def"]:
        invalid = per_game[f"{side}_drives"].lt(5)
        for metric in ["ppd", "start"]:
            per_game.loc[invalid, f"drive_{side}_{metric}"] = np.nan

    order = games[["game_id", "season", "date", "home_team", "away_team",
                   "home_points", "away_points"]].copy()
    home = order.rename(columns={"home_team": "team", "away_team": "opponent"})
    home["home"] = True
    away = order.rename(columns={"away_team": "team", "home_team": "opponent"})
    away["home"] = False
    long = pd.concat([home, away], ignore_index=True).merge(
        per_game, on=["game_id", "team"], how="left")
    long = long.sort_values(["team", "date", "game_id"]).reset_index(drop=True)
    for metric in DRIVE_COLS:
        # Scheduled games and the current game's drive summary are excluded.
        observed = long[metric].where(long.home_points.notna() & long.away_points.notna())
        earlier = observed.groupby([long.team, long.season]).shift(1)
        grouped = [long.team, long.season]
        recent = (earlier.groupby(grouped).rolling(8, min_periods=1).mean()
                  .reset_index(level=[0, 1], drop=True))
        count = (earlier.groupby(grouped).rolling(8, min_periods=1).count()
                 .reset_index(level=[0, 1], drop=True).fillna(0))
        previous = (long.assign(observed=observed).dropna(subset=["observed"])
                    .groupby(["team", "season"])["observed"]
                    .apply(lambda values: values.tail(8).mean()).to_dict())
        prior = pd.Series([previous.get((team, season-1), np.nan)
                           for team, season in zip(long.team, long.season)], index=long.index)
        weight = (2 * (1 - count / 8)).clip(lower=0).where(prior.notna(), 0)
        long[metric + "_prior"] = ((recent.fillna(0) * count + prior.fillna(0) * weight) /
                                   (count + weight)).where((count + weight) > 0)
    keep = ["game_id"] + [m + "_prior" for m in DRIVE_COLS]
    h = long[long.home][keep].add_prefix("home_").rename(columns={"home_game_id": "game_id"})
    a = long[~long.home][keep].add_prefix("away_").rename(columns={"away_game_id": "game_id"})
    result = games.merge(h, on="game_id", how="left", validate="one_to_one")
    result = result.merge(a, on="game_id", how="left", validate="one_to_one")
    result["drive_ppd_matchup_home"] = (result.home_drive_off_ppd_prior -
                                        result.away_drive_def_ppd_prior)
    result["drive_ppd_matchup_away"] = (result.away_drive_off_ppd_prior -
                                        result.home_drive_def_ppd_prior)
    # Lower yards-to-goal means better starting field position.
    result["drive_field_position_diff"] = (result.away_drive_off_start_prior -
                                            result.home_drive_off_start_prior)
    return result


def audit(result):
    base = heldout(result)
    candidate = heldout(result, model_factory=lambda: V1Model(
        features=schema.MODEL_FEATURES + NEW_FEATURES))
    b = base[["game_id", "p_model"]].rename(columns={"p_model": "p_base"})
    c = candidate[["game_id", "p_model"]].rename(columns={"p_model": "p_drive"})
    scored = result.merge(b,on="game_id",validate="one_to_one").merge(
        c,on="game_id",validate="one_to_one")
    scored = scored[scored.home_classification.str.lower().eq("fbs") &
                    scored.away_classification.str.lower().eq("fbs")].copy()
    y = scored.home_points.gt(scored.away_points).to_numpy()
    for label, data in [("all", scored), *[(str(y), d) for y,d in scored.groupby("season")]]:
        actual = data.home_points.gt(data.away_points).to_numpy()
        print(f"{label}: n={len(data)} drive coverage="
              f"{data[NEW_FEATURES].notna().all(axis=1).mean():.1%}")
        for name in ["p_base", "p_drive", "market_home_prob"]:
            p = data[name].to_numpy(dtype=float)
            valid = np.isfinite(p)
            print(f"  {name}: n={valid.sum()} accuracy={((p[valid]>=.5)==actual[valid]).mean():.4f} "
                  f"Brier={np.mean((p[valid]-actual[valid])**2):.5f}")
    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--seasons", default="2014-2026")
    parser.add_argument("--training", type=Path, default=Path("data/derived/training.parquet"))
    parser.add_argument("--drives", type=Path, default=Path("data/raw/drives_raw.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/drive_audit.parquet"))
    args = parser.parse_args()
    drives = fetch_drives(_season_list(args.seasons), args.drives) if args.fetch else pd.read_parquet(args.drives)
    result = drive_features(pd.read_parquet(args.training), drives)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit(result).to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
