"""Compare frozen baseline and corrected profiles on later seasons, plus pool slates."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from backtest import _metrics
from model import V1Model
import features
import schema


def main():
    spec = importlib.util.spec_from_file_location("original_features", "data/original_features.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    base = pd.read_parquet("data/derived/games_base.parquet")
    adv = pd.read_parquet("data/raw/advanced_raw.parquet")
    old = original.build(base, adv)
    new = features.build(base, adv)
    new.to_parquet("data/derived/training.parquet", index=False)
    outputs = []
    for season in sorted(new.season.unique())[3:]:
        test = new[(new.season == season) & new.home_points.notna() & new.away_points.notna()].copy()
        test = test[test.home_classification.str.lower().eq("fbs") & test.away_classification.str.lower().eq("fbs")]
        if len(test) < 50:
            continue
        for name, data in [("original", old), ("corrected", new)]:
            train = data[(data.season < season) & data.home_points.notna() & data.away_points.notna()]
            model = V1Model().fit(train)
            test["p_" + name] = model.predict_proba(data.set_index("game_id").loc[test.game_id].reset_index())
        train = new[(new.season < season) & new.home_points.notna() & new.away_points.notna()]
        train = train[train.home_classification.str.lower().eq("fbs") & train.away_classification.str.lower().eq("fbs")]
        for name, learner in [("logistic", LogisticRegression(C=0.1, max_iter=2000)),
                              ("neighbors", KNeighborsClassifier(n_neighbors=75))]:
            model = make_pipeline(SimpleImputer(add_indicator=True, keep_empty_features=True), StandardScaler(), learner)
            model.fit(train[schema.MODEL_FEATURES], train.home_points.gt(train.away_points))
            test["p_" + name] = model.predict_proba(test[schema.MODEL_FEATURES])[:, 1]
        outputs.append(test)
        print(f"Finished held-out season {season}", flush=True)
    out = pd.concat(outputs, ignore_index=True)
    out.to_parquet("data/derived/profile_oos.parquet", index=False)
    records = []
    for label, subset in [("all", out), ("weeks_1_4", out[out.week.le(4)]),
                          ("2023_2025", out[out.season.between(2023, 2025)])]:
        for name in ("original", "corrected", "logistic", "neighbors"):
            records.append({"subset": label, "model": name,
                            **_metrics(subset.home_points.gt(subset.away_points), subset["p_" + name])})
    print(pd.DataFrame(records).to_string(index=False))
    # Archived pool membership defines the evaluation subset. These are
    # season-held-out replays, not the exact probabilities published that week.
    ids = set()
    for path in Path('historicals/predictions').glob('*.json'):
        archive = json.loads(path.read_text())
        for game in archive['games']:
            if game.get('game_id'):
                ids.add(str(game['game_id']))
                continue
            # Legacy archives omitted IDs; resolve only unique season/week
            # schedule matches. Do not invent a game ID or broaden the slate.
            matches = new[(new.season == archive['season']) & (new.week == archive['week']) &
                          (((new.home_team == game['home_team']) & (new.away_team == game['away_team'])) |
                           ((new.home_team == game['away_team']) & (new.away_team == game['home_team'])))]
            if len(matches) == 1:
                ids.add(str(matches.iloc[0].game_id))
    pool = out[out.game_id.astype(str).isin(ids)]
    for name in ("original", "corrected", "logistic", "neighbors"):
        print("archived pool replay", name, _metrics(pool.home_points.gt(pool.away_points), pool['p_' + name]))
    Path("data/derived/profile_metrics.json").write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
