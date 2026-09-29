"""Walk-forward experiment: do similar pregame underdogs reveal hidden winners?

Research-only diagnostic. The line is a benchmark/context for defining an
underdog; no future-season outcome enters either a neighbor or calibration.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss
from audit_fbs import heldout

DATA = "data/derived/training.parquet"


def orient(g):
    d = g[g.spread_home.notna() & g.spread_home.ne(0)].copy()
    h = d.spread_home.gt(0)
    d["dog_win"] = d.home_points.gt(d.away_points).eq(h).astype(int)
    d["dog_at_home"] = h.astype(int)
    d["dog_elo"] = np.where(h, d.elo_home_prob, 1 - d.elo_home_prob)
    d["dog_rest"] = np.where(h, d.rest_diff, -d.rest_diff)
    d["dog_model_prob"] = np.where(h, d.p_model, 1 - d.p_model)
    for metric in ("off_ppa_adj", "def_ppa_adj", "off_success_adj",
                   "off_explosive_adj"):
        d["dog_" + metric] = np.where(h, d["home_" + metric], d["away_" + metric])
        d["fav_" + metric] = np.where(h, d["away_" + metric], d["home_" + metric])
    d["line_abs"] = d.spread_home.abs()
    return d


FEATURES = ["dog_off_ppa_adj", "fav_def_ppa_adj", "fav_off_ppa_adj",
            "dog_def_ppa_adj", "dog_off_success_adj", "fav_off_success_adj",
            "dog_off_explosive_adj", "fav_off_explosive_adj", "dog_elo",
            "dog_rest", "dog_at_home", "neutral_site"]


def run():
    held = heldout(pd.read_parquet(DATA))
    held = held[held.home_classification.str.lower().eq("fbs") &
                held.away_classification.str.lower().eq("fbs")]
    d = orient(held)
    rows = []
    for season in sorted(d.season.unique()):
        if season < 2020:
            continue  # 2017–2019 supply earlier-season training analogs.
        train = d[d.season < season].copy()
        test = d[d.season == season].copy()
        line = LogisticRegression().fit(train[["line_abs"]], train.dog_win)
        train["p_line"] = line.predict_proba(train[["line_abs"]])[:, 1]
        test["p_line"] = line.predict_proba(test[["line_abs"]])[:, 1]
        for name, cols in (("profile", FEATURES), ("profile_line", FEATURES + ["line_abs"])):
            imp = SimpleImputer(strategy="median", add_indicator=False)
            scaler = StandardScaler()
            a = scaler.fit_transform(imp.fit_transform(train[cols]))
            b = scaler.transform(imp.transform(test[cols]))
            neighbors = NearestNeighbors(n_neighbors=75).fit(a).kneighbors(b, return_distance=False)
            residuals = (train.dog_win - train.p_line).to_numpy()[neighbors]
            test["p_" + name] = np.clip(test.p_line + residuals.mean(axis=1), .01, .99)
            test["edge_" + name] = residuals.mean(axis=1)
            # Similar historical outcomes describe the neighborhood; there
            # are 75 analogs, not an exact copy of the upcoming matchup.
            test["neighbor_wins_" + name] = train.dog_win.to_numpy()[neighbors].sum(axis=1)
        rows.append(test)
    out = pd.concat(rows, ignore_index=True)
    for name in ("line", "profile", "profile_line"):
        p = out["p_" + name]
        print(name, "n", len(out), "Brier", round(brier_score_loss(out.dog_win, p), 4),
              "logloss", round(log_loss(out.dog_win, p), 4))
        print("  by season:", [(int(s), round(brier_score_loss(sub.dog_win, sub["p_" + name]), 4))
                               for s, sub in out.groupby("season")])
    for name in ("profile", "profile_line"):
        selected = out[out["edge_" + name] >= out["edge_" + name].quantile(.9)]
        print(name, "top 10% positive residual", len(selected),
              "upsets", round(selected.dog_win.mean(), 3),
              "line expectation", round(selected.p_line.mean(), 3),
              "model dog probability", round(selected.dog_model_prob.mean(), 3))
        disagree = out[out.dog_model_prob.gt(.5)]
        print(name, "model-chosen dogs Brier", round(brier_score_loss(disagree.dog_win,
                                                          disagree["p_" + name]), 4))
    out.to_parquet("data/derived/upset_analogs.parquet", index=False)


if __name__ == "__main__":
    run()
