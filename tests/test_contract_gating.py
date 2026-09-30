import unittest
import numpy as np
import pandas as pd
import schema
from contract import validate, DataContractError

class ContractGatingTests(unittest.TestCase):
    def table(self):
        d = pd.DataFrame({c:[.1,.2] for c in schema.ALL_COLS})
        for c in ['game_id','home_team','away_team']:
            d[c] = ['a','b']
        d['season'] = 2026; d['week'] = [1,4]
        d['neutral_site'] = [True,False]
        d['home_classification'] = 'fbs';d['away_classification'] = 'fbs'
        d['home_prior_games'] = [0,3];d['away_prior_games'] = [0,3]
        for c in schema.STRENGTH_FEATURES:
            if c != 'elo_home_prob':
                d.loc[0,c] = np.nan
        return d

    def test_deliberate_early_gating_is_allowed(self):
        d = self.table()
        # Constant-feature check needs multiple eligible values, so supply a
        # second mature game with a distinct observed efficiency profile.
        extra = d.iloc[[1]].copy();extra['game_id']='c'
        for c in schema.STRENGTH_FEATURES:
            extra[c] = .3
        d = pd.concat([d,extra],ignore_index=True)
        validate(d,strength_history_min=3)

    def test_missing_mature_efficiency_is_rejected(self):
        d = self.table();d.loc[1,'off_ppa_adj_diff'] = np.nan
        with self.assertRaises(DataContractError):
            validate(d,strength_history_min=3)

    def test_feed_outage_cannot_be_hidden_by_gating(self):
        d = self.table();d['week']=[4,5]
        d['home_prior_games']=0;d['away_prior_games']=0
        with self.assertRaisesRegex(DataContractError,'no history-eligible'):
            validate(d,strength_history_min=3)
