import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from model import StatisticalModel
import schema

class StatisticalModelTests(unittest.TestCase):
    def test_terms_reconstruct_probability_and_market_is_excluded(self):
        rng = np.random.default_rng(42)
        d = pd.DataFrame(rng.normal(size=(200,len(schema.MODEL_FEATURES))),columns=schema.MODEL_FEATURES)
        d['home_points'] = np.where(d.off_ppa_adj_diff > 0,28,7)
        d['away_points'] = 14
        d['home_classification'] = 'fbs';d['away_classification'] = 'fbs'
        d.loc[:20,'off_ppa_adj_diff'] = np.nan
        d['spread_home'] = rng.normal(size=200)
        model = StatisticalModel().fit(d)
        p = model.predict_proba(d)
        intercept = model.pipeline.steps[-1][1].intercept_[0]
        np.testing.assert_allclose(expit(model.contributions(d).sum(axis=1)+intercept),p)
        d['spread_home'] = 9999
        np.testing.assert_allclose(model.predict_proba(d),p)
