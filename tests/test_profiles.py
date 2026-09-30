import unittest
import numpy as np
import pandas as pd
from features import _team_game_long, _opponent_adjust
from profile_analogs import ProfileAnalogs, PROFILE_FEATURES
from predict import find_game


class ProfileTests(unittest.TestCase):
    def test_future_schedule_does_not_evict_observed_form(self):
        rows, stats = [], []
        for i in range(20):
            rows.append(dict(game_id=str(i), season=2026, week=i+1,
                             date=pd.Timestamp('2026-08-01', tz='UTC') + pd.Timedelta(days=i*7),
                             home_team='A', away_team='B', neutral_site=False,
                             home_points=21 if i < 4 else np.nan,
                             away_points=7 if i < 4 else np.nan))
            if i < 4:
                for team in ['A', 'B']:
                    stats.append(dict(game_id=str(i), team=team, **{
                        m: float(i+1) for m in ['off_ppa','def_ppa','off_success',
                                                'def_success','off_explosive','def_explosive']}))
        tg = _team_game_long(pd.DataFrame(rows), pd.DataFrame(stats))
        future = tg[(tg.team == 'A') & tg.game_id.astype(int).ge(4)]
        self.assertTrue(future.off_ppa_roll.eq(2.5).all())
        self.assertTrue(future.prior_games.eq(4).all())

    def test_current_game_stats_cannot_change_its_pregame_form(self):
        rows = pd.DataFrame([dict(game_id=str(i), season=2026, week=i+1,
                             date=pd.Timestamp('2026-08-01',tz='UTC')+pd.Timedelta(days=i*7),
                             home_team='A',away_team='B',neutral_site=False,
                             home_points=21,away_points=7) for i in range(5)])
        metrics = ['off_ppa','def_ppa','off_success','def_success','off_explosive','def_explosive']
        stats = pd.DataFrame([dict(game_id=str(i),team=t,**{m:.4 for m in metrics})
                              for i in range(5) for t in ['A','B']])
        before = _team_game_long(rows,stats)
        stats.loc[stats.game_id.eq('3'),metrics] = 999
        after = _team_game_long(rows,stats)
        cols = ['game_id','team','prior_games']+[m+'_roll' for m in metrics]
        pd.testing.assert_frame_equal(before[before.game_id.le('3')][cols],
                                      after[after.game_id.le('3')][cols])

    def test_opponent_window_resets_with_season(self):
        tg = pd.DataFrame([
            dict(game_id=str(i), team=t, opponent='B' if t=='A' else 'A',
                 season=2025 if i < 2 else 2026, date=pd.Timestamp('2025-09-01', tz='UTC')+pd.Timedelta(days=7*i),
                 is_home=t=='A', home_points=21, away_points=7,
                 off_ppa_roll=.4, def_ppa_roll=(-.5 if t=='B' else .9))
            for i in range(3) for t in ['A','B']])
        got = _opponent_adjust(tg, 'off_ppa_roll', 'def_ppa_roll')
        self.assertEqual(got.iloc[4], .4)

    def test_analogs_exclude_current_season_and_ignore_team_identity(self):
        h = pd.DataFrame([dict(game_id=str(i), season=2024 if i<80 else 2026,
                     date=pd.Timestamp('2024-09-01',tz='UTC'), home_team='Old A',away_team='Old B',
                     home_classification='fbs', away_classification='fbs', neutral_site=False,
                     home_points=21 if i<80 else 0,away_points=7,
                     **{c: float(i%5) for c in PROFILE_FEATURES}) for i in range(100)])
        row = h.iloc[0].copy();row['season']=2026;row['date']=pd.Timestamp('2026-10-01',tz='UTC')
        a = ProfileAnalogs(h).describe(row)
        self.assertEqual(a['n'],75);self.assertEqual(a['home_wins'],75)
        row['home_team']='New A';row['away_team']='New B'
        self.assertEqual(a,ProfileAnalogs(h).describe(row))
        row[PROFILE_FEATURES[0]]=np.nan
        self.assertFalse(ProfileAnalogs(h).describe(row)['available'])

    def test_schedule_lookup_does_not_repredict_completed_matchup(self):
        h = pd.DataFrame([dict(home_team='A',away_team='B',home_points=21,
                              date=pd.Timestamp('2025-10-01',tz='UTC'))])
        self.assertEqual(find_game(h,'B','A'),(None,False))


if __name__ == '__main__':
    unittest.main()
