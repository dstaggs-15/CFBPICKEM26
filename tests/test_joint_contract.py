import unittest
import numpy as np
import pandas as pd
import schema
from contract import validate_joint, DataContractError


class JointContractTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(19)
        d=pd.DataFrame(rng.normal(size=(30,len(schema.JOINT_MODEL_FEATURES))),columns=schema.JOINT_MODEL_FEATURES)
        d['season']=2026;d['week']=6;d['joint_min_games']=3
        return d

    def test_missing_mature_stat_feed_is_rejected(self):
        d=self.fixture();d['joint_off_ppa_edge']=np.nan
        with self.assertRaises(DataContractError):
            validate_joint(d)

    def test_early_gating_is_allowed_but_missing_elo_is_rejected(self):
        d=self.fixture();d.loc[:4,'joint_min_games']=0
        d.loc[:4,schema.JOINT_PROFILE_FEATURES]=np.nan
        validate_joint(d)
        d['elo_home_prob']=np.nan
        with self.assertRaises(DataContractError):
            validate_joint(d)
