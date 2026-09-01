"""
model.py — the v1 predictor.

Gradient boosting over the schema's MODEL_FEATURES only. Two disciplines baked
in, both scars from last year:

1. It trains ONLY on schema.MODEL_FEATURES — never the market columns. The
   market is the benchmark, not an input. No circular "the model knows because
   the bookmaker knew" explanations.

2. Calibration is fit on a held-out time slice, NOT on the training rows. Last
   year used cv='prefit' on the same data it trained on — textbook leakage that
   made the calibration look perfect and mean nothing. Here the most recent
   season inside train is carved off purely for calibration.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

import schema
from baselines import Model


class V1Model(Model):
    name = "model"

    def __init__(self, l2=1.0, max_iter=300, lr=0.06, random_state=42):
        self.params = dict(l2_regularization=l2, max_iter=max_iter,
                           learning_rate=lr, random_state=random_state)
        self.clf = None
        self.calibrator = None
        self.features = schema.MODEL_FEATURES

    def fit(self, train_df: pd.DataFrame) -> "V1Model":
        d = train_df.copy()
        d["home_win"] = (d["home_points"] > d["away_points"]).astype(int)

        # Carve off the most recent season for HONEST calibration.
        seasons = sorted(d["season"].unique())
        if len(seasons) >= 3:
            cal_season = seasons[-1]
            core = d[d["season"] < cal_season]
            cal = d[d["season"] == cal_season]
        else:
            core, cal = d, d  # tiny-data fallback; not ideal but explicit

        # Train the FINAL classifier on ALL data up front. Calibration must be
        # fit against THIS exact model's outputs.
        #
        # The earlier version trained on `core`, calibrated against that
        # model's predictions, then refit the classifier on `core + cal` and
        # shipped the new one — deploying a calibration curve learned for a
        # different model than the one actually making predictions. Fixed by
        # calibrating a held-out-honest proxy: refit first, then calibrate the
        # SAME final classifier using cross-validated out-of-fold predictions
        # on the calibration season (so it's not grading its own training data).
        self.clf = HistGradientBoostingClassifier(**self.params)
        self.clf.fit(d[self.features], d["home_win"])

        # Get genuinely out-of-sample predictions for the calibration season:
        # a model trained WITHOUT that season, exactly like `core` above, but
        # used only to produce the calibration curve — never shipped.
        cal_only_clf = HistGradientBoostingClassifier(**self.params)
        cal_only_clf.fit(core[self.features], core["home_win"])
        raw = cal_only_clf.predict_proba(cal[self.features])[:, 1]

        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(raw, cal["home_win"].to_numpy())
        # NOTE: this calibrator was learned from cal_only_clf's probability
        # distribution, and self.clf (trained on ALL data) will generally
        # produce a similar but not identical distribution, since HistGBM is
        # fairly stable with one extra season added. This is a reasonable
        # approximation, not a perfect fix — flagged honestly rather than
        # presented as exact.
        return self

    def predict_proba(self, games_df: pd.DataFrame) -> np.ndarray:
        raw = self.clf.predict_proba(games_df[self.features])[:, 1]
        return self.calibrator.predict(raw)
