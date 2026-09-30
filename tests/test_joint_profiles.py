import unittest
import numpy as np
import pandas as pd
from joint_profile_audit import build_profiles, PROFILE


class JointProfileTests(unittest.TestCase):
    def fixture(self):
        games = pd.DataFrame([dict(game_id=str(i),season=2026,week=i+1,
            date=pd.Timestamp('2026-08-29',tz='UTC')+pd.Timedelta(days=7*i),
            home_team='A',away_team='B',home_classification='fbs',away_classification='fbs',
            neutral_site=False,home_points=28.,away_points=14.) for i in range(5)])
        advanced = pd.DataFrame([dict(game_id=str(i),team=t,off_ppa=.3 if t=='A' else .1,
            off_success=.5,off_explosive=1.2) for i in range(5) for t in ['A','B']])
        return games,advanced

    def test_future_and_same_week_results_cannot_change_snapshot(self):
        games,advanced = self.fixture()
        before = build_profiles(games,advanced,4.)
        games.loc[games.game_id.ge('3'),['home_points','away_points']] = [0.,99.]
        advanced.loc[advanced.game_id.ge('3'),'off_ppa'] = 99.
        after = build_profiles(games,advanced,4.)
        pd.testing.assert_frame_equal(before.loc[before.game_id.le('3'),PROFILE],
                                      after.loc[after.game_id.le('3'),PROFILE])

    def test_names_and_previous_season_have_no_rating_effect(self):
        games,advanced = self.fixture()
        before = build_profiles(games,advanced,4.)
        old = games.copy();old['season']=2025;old['date']-=pd.Timedelta(days=364)
        old['home_points']=99.;old['away_points']=0.
        expanded = pd.concat([old,games],ignore_index=True)
        after = build_profiles(expanded,advanced,4.).tail(len(games)).reset_index(drop=True)
        pd.testing.assert_frame_equal(before[PROFILE],after[PROFILE])
        games['home_team']='Z';games['away_team']='Y'
        advanced['team']=advanced.team.map({'A':'Z','B':'Y'})
        renamed = build_profiles(games,advanced,4.)
        np.testing.assert_allclose(before[PROFILE],renamed[PROFILE],equal_nan=True)
