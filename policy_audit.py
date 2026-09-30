"""Evaluate the live probability policy using prior-season-only forecasts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_fbs import heldout
from prediction_policy import combine, POLICY


def main():
    data = heldout(pd.read_parquet("data/derived/training.parquet"))
    data = data[data.home_classification.str.lower().eq("fbs") &
                data.away_classification.str.lower().eq("fbs") &
                data.spread_home.notna() & data.spread_home.ne(0)].copy()
    data["p_final"] = [combine(a, b)[0] for a,b in zip(data.p_model,data.market_home_prob)]
    y = data.home_points.gt(data.away_points)
    rows = []
    groups = [("all", data), ("completed 2021–25", data[data.season.between(2021,2025)]),
              ("weeks 1–4", data[data.week.le(4)]),
              ("neutral site", data[data.neutral_site]),
              ("spread 0–3", data[data.spread_home.abs().le(3)]),
              ("spread 3–7", data[data.spread_home.abs().gt(3)&data.spread_home.abs().le(7)]),
              ("spread >7", data[data.spread_home.abs().gt(7)]),
              ("final underdog picks", data[data.p_final.ge(.5).ne(data.spread_home.le(0))])]
    for label, group in groups:
        if group.empty:
            continue
        actual = group.home_points.gt(group.away_points)
        row = {"group":label,"games":len(group),
               "policy_accuracy":float(group.p_final.ge(.5).eq(actual).mean()),
               "independent_accuracy":float(group.p_model.ge(.5).eq(actual).mean()),
               "favorite_accuracy":float(group.spread_home.le(0).eq(actual).mean()),
               "brier":float(np.mean((group.p_final-actual.astype(int))**2))}
        rows.append(row)
        print(json.dumps(row),flush=True)
    # Stop a release if it fails to beat the current independent model across
    # the completed-season comparison. This is a release guard, not proof of
    # a market edge or future accuracy guarantee.
    completed = next(r for r in rows if r["group"] == "completed 2021–25")
    if completed["policy_accuracy"] <= completed["independent_accuracy"]:
        raise RuntimeError("Candidate did not improve completed-season accuracy")
    Path("data/derived/policy_audit.json").write_text(json.dumps({
        "policy":POLICY,"notes":"Ranking nudge excluded; saved historical lines may differ from weekly deadline quotes; 2026 partial.",
        "groups":rows},indent=2)+"\n")


if __name__ == "__main__":
    main()
