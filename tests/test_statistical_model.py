import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from model import StatisticalModel, JointStatisticalModel
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

    def test_joint_model_uses_both_stats_and_elo_in_one_fitted_probability(self):
        rng = np.random.default_rng(17)
        d = pd.DataFrame(rng.normal(size=(500,len(schema.JOINT_MODEL_FEATURES))),columns=schema.JOINT_MODEL_FEATURES)
        d['elo_home_prob'] = rng.uniform(.1,.9,len(d))
        score = d.joint_points_edge + 3*(d.elo_home_prob-.5) + rng.normal(size=len(d))
        d['home_points'] = np.where(score > 0,28,7);d['away_points']=14
        d['home_classification']='fbs';d['away_classification']='fbs'
        model=JointStatisticalModel().fit(d)
        row=d.iloc[[0]].copy();original=model.predict_proba(row)[0]
        row['joint_points_edge']+=1
        self.assertGreater(model.predict_proba(row)[0],original)
        row=d.iloc[[0]].copy();row['elo_home_prob']+=.1
        self.assertGreater(model.predict_proba(row)[0],original)
        intercept=model.pipeline.steps[-1][1].intercept_[0]
        np.testing.assert_allclose(expit(model.contributions(d).sum(axis=1)+intercept),model.predict_proba(d))
        self.assertEqual(model.features,schema.JOINT_MODEL_FEATURES)
        altered=d.copy();altered['home_team']='Alabama';altered['spread_home']=-99
        np.testing.assert_allclose(model.predict_proba(altered),model.predict_proba(d))
