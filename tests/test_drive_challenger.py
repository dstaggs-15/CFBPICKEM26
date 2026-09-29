import unittest

import pandas as pd

from drive_challenger import drive_features, parse_drives


class DriveChallengerTests(unittest.TestCase):
    def test_current_and_future_drives_do_not_enter_pregame_features(self):
        games = pd.DataFrame([
            dict(game_id=str(i), season=2025, date=pd.Timestamp('2025-09-01', tz='UTC') +
                 pd.Timedelta(days=7*i), home_team='A', away_team='B',
                 home_points=21, away_points=7)
            for i in range(3)
        ])
        drives = pd.DataFrame([
            dict(game_id=str(i), drive_id=f'{i}-{team}-{j}', offense=team,
                 defense='B' if team == 'A' else 'A', start_yards_to_goal=80,
                 points=points)
            for i, points in enumerate([1, 2, 8])
            for team in ['A', 'B'] for j in range(6)
        ])
        result = drive_features(games, drives)
        self.assertTrue(pd.isna(result.iloc[0].home_drive_off_ppd_prior))
        self.assertEqual(result.iloc[1].home_drive_off_ppd_prior, 1)
        self.assertEqual(result.iloc[2].home_drive_off_ppd_prior, 1.5)

    def test_drive_parser_ignores_overtime_and_duplicate_ids(self):
        def row(ident, period):
            return dict(id=ident, gameId=99, offense='A', defense='B',
                        startPeriod=period, startYardsToGoal=75,
                        startOffenseScore=7, endOffenseScore=10)
        got = parse_drives([row('x', 1), row('x', 1), row('ot', 5)])
        self.assertEqual(len(got), 1)
        self.assertEqual(got.iloc[0].points, 3)


if __name__ == '__main__':
    unittest.main()
