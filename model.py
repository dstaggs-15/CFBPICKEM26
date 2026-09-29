"""
model.py — the v1 predictor.

Gradient boosting over the schema's MODEL_FEATURES only. Two disciplines baked
in, both scars from last year:

1. It trains ONLY on schema.MODEL_FEATURES — never the market columns. The
   market is the benchmark, not an input. No circular "the model knows because
   the bookmaker knew" explanations.

2. Calibration is fit on predictions for a held-out season from a classifier
   trained only on earlier seasons. A sigmoid calibrator beat isotonic on
   held-out FBS probability scores; it is the default until retested.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import logit

import schema
from baselines import Model


class V1Model(Model):
    name = "model"

    def __init__(self, l2=1.0, max_iter=300, lr=0.06, random_state=42,
                 features=None, calibration="sigmoid"):
        self.params = dict(l2_regularization=l2, max_iter=max_iter,
                           learning_rate=lr, random_state=random_state)
        self.clf = None
        self.calibrator = None
        self.features = list(features) if features is not None else schema.MODEL_FEATURES
        if calibration not in ("isotonic", "sigmoid", "none"):
            raise ValueError("calibration must be isotonic, sigmoid, or none")
        self.calibration = calibration

    def fit(self, train_df: pd.DataFrame) -> "V1Model":
        # Stable tie-breaking matters when many Saturday kickoffs share a time.
        d = train_df.sort_values(["season", "week", "date", "game_id"]).copy()
        d["home_win"] = (d["home_points"] > d["away_points"]).astype(int)

        # Carve off the most recent season for HONEST calibration.
        seasons = sorted(d["season"].unique())
        # Ship a classifier trained on all earlier seasons. The calibration
        # curve below comes from a proxy that has not seen the calibration
        # season, so those probability/label pairs are out-of-sample.
        self.clf = HistGradientBoostingClassifier(**self.params)
        self.clf.fit(d[self.features], d["home_win"])
        if len(seasons) < 3 or self.calibration == "none":
            # No independent calibration season in tiny datasets; return raw
            # scores instead of fitting an in-sample calibration curve.
            return self
        cal_season = seasons[-1]
        core = d[d["season"] < cal_season]
        cal = d[d["season"] == cal_season]

        # The proxy is trained WITHOUT the calibration season and used only to
        # produce the calibration curve. It is not the shipped classifier.
        cal_only_clf = HistGradientBoostingClassifier(**self.params)
        cal_only_clf.fit(core[self.features], core["home_win"])
        raw = cal_only_clf.predict_proba(cal[self.features])[:, 1]

        if self.calibration == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw, cal["home_win"].to_numpy())
        elif self.calibration == "sigmoid":
            self.calibrator = LogisticRegression()
            self.calibrator.fit(logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1),
                                cal["home_win"].to_numpy())
        # The final classifier also saw the calibration season, so its raw
        # score distribution can shift. This is an approximation; the outer
        # walk-forward test measures whether it helps on later seasons.
        return self

    def predict_proba(self, games_df: pd.DataFrame) -> np.ndarray:
        raw = self.clf.predict_proba(games_df[self.features])[:, 1]
        if self.calibrator is None:
            return raw
        if self.calibration == "sigmoid":
            return self.calibrator.predict_proba(
                logit(np.clip(raw, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]
        return self.calibrator.predict(raw)
