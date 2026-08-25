"""
team_stats.py — per-team, current-season stat lines with league ranks.

Powers the dropdown on each game card ("#5 defense", etc.). Built from the
advanced stats we already fetch at 100% coverage:
    offensive efficiency (PPA), defensive efficiency (PPA),
    success rate, explosiveness.

If a season counting-stats file exists (data/raw/season_stats_raw.parquet, e.g.
turnovers), those get merged and ranked too — otherwise we simply show the
efficiency ranks we can stand behind. Nothing invented.

Ranks are 1 = best. For defense, lower PPA allowed is better, so it's ranked
ascending; offense/explosiveness/success are ranked descending.
"""

from __future__ import annotations
import pandas as pd

ADV_PARQUET = "data/raw/advanced_raw.parquet"
BASE_PARQUET = "data/derived/games_base.parquet"
SEASON_STATS = "data/raw/season_stats_raw.parquet"  # optional
OUT_JSON = "docs/team_stats.json"


def _rank(series: pd.Series, ascending: bool) -> pd.Series:
    return series.rank(ascending=ascending, method="min").astype("Int64")


def build_for_season(season: int) -> dict:
    adv = pd.read_parquet(ADV_PARQUET)
    base = pd.read_parquet(BASE_PARQUET)[["game_id", "season"]]
    adv = adv.merge(base, on="game_id", how="left")
    adv = adv[adv["season"] == season]
    if adv.empty:
        return {}

    agg = adv.groupby("team").agg(
        off_ppa=("off_ppa", "mean"),
        def_ppa=("def_ppa", "mean"),
        success=("off_success", "mean"),
        explosive=("off_explosive", "mean"),
        games=("game_id", "nunique"),
    ).reset_index()

    n = len(agg)
    agg["off_rank"] = _rank(agg["off_ppa"], ascending=False)
    agg["def_rank"] = _rank(agg["def_ppa"], ascending=True)   # lower is better
    agg["success_rank"] = _rank(agg["success"], ascending=False)
    agg["explosive_rank"] = _rank(agg["explosive"], ascending=False)

    # optional counting stats (turnovers, etc.)
    extra = {}
    try:
        ss = pd.read_parquet(SEASON_STATS)
        ss = ss[ss["season"] == season]
        # expected long format: team, stat_name, stat_value
        piv = ss.pivot_table(index="team", columns="stat_name",
                             values="stat_value", aggfunc="first")
        for stat in piv.columns:
            asc = stat.lower() in ("turnovers", "turnoverslost", "interceptionsthrown", "fumberslost")
            piv[f"{stat}__rank"] = _rank(piv[stat], ascending=asc)
        extra = piv.to_dict("index")
    except Exception:
        pass  # no season counting stats available; efficiency ranks only

    out = {}
    for _, r in agg.iterrows():
        stats = [
            {"label": "Offense (PPA/play)", "value": round(r.off_ppa, 3), "rank": int(r.off_rank), "of": n},
            {"label": "Defense (PPA allowed)", "value": round(r.def_ppa, 3), "rank": int(r.def_rank), "of": n},
            {"label": "Success rate", "value": f"{r.success:.1%}", "rank": int(r.success_rank), "of": n},
            {"label": "Explosiveness", "value": round(r.explosive, 2), "rank": int(r.explosive_rank), "of": n},
        ]
        team_extra = extra.get(r.team, {})
        for k, v in team_extra.items():
            if k.endswith("__rank"):
                continue
            rk = team_extra.get(f"{k}__rank")
            stats.append({"label": k, "value": v,
                          "rank": int(rk) if pd.notna(rk) else None, "of": n})
        out[r.team] = {"games": int(r.games), "stats": stats}
    return out


if __name__ == "__main__":
    import json, pandas as pd
    base = pd.read_parquet(BASE_PARQUET)
    season = int(base["season"].max())
    data = build_for_season(season)
    with open(OUT_JSON, "w") as f:
        json.dump({"season": season, "teams": data}, f, indent=2)
    print(f"Wrote {OUT_JSON} for {len(data)} teams (season {season}).")
