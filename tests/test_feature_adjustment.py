import unittest

import numpy as np
import pandas as pd

from features import _elo_probs, _opponent_adjust, _team_game_long
from ranking_signal import adjustment
from fetch_cfbd import parse_advanced


class FeatureAdjustmentTests(unittest.TestCase):
    def test_passing_and_rushing_stats_use_correct_units(self):
        game = {"gameId": 1, "team": "A", "offense": {
            "passingPlays": {"ppa": 0.4}, "rushingPlays": {"ppa": -0.1}},
            "defense": {"passingPlays": {"ppa": 0.2},
                        "rushingPlays": {"ppa": -0.3}}}
        parsed = parse_advanced([game]).iloc[0]
        self.assertEqual(parsed.off_pass_ppa, 0.4)
        self.assertEqual(parsed.off_rush_ppa, -0.1)
        self.assertEqual(parsed.def_pass_ppa, 0.2)
        self.assertEqual(parsed.def_rush_ppa, -0.3)

    def test_offense_faces_defense_and_ignores_future_rows(self):
        # Team A faces a strong defense (low defensive PPA). Team B's offense
        # is deliberately high: using the wrong side would reverse the boost.
        rows = [
            ("g0", "Other", "Weak", "2025-09-01", 0.4, 0.3, 1, 0),
            ("g0", "Weak", "Other", "2025-09-01", 0.9, 0.3, 1, 0),
            ("g1", "A", "B", "2025-09-08", 0.4, 0.8, 1, 0),
            ("g1", "B", "A", "2025-09-08", 0.9, -0.2, 1, 0),
            ("g2", "A", "C", "2025-09-15", 0.4, 0.8, 1, 0),
            ("g2", "C", "A", "2025-09-15", 0.9, 0.3, 1, 0),
        ]
        tg = pd.DataFrame(rows, columns=["game_id", "team", "opponent", "date",
                                         "off_ppa_roll", "def_ppa_roll", "home_points", "away_points"])
        tg["date"] = pd.to_datetime(tg["date"], utc=True)
        tg["is_home"] = tg["team"].isin(["Other", "A"])
        adjusted = _opponent_adjust(tg, "off_ppa_roll", "def_ppa_roll")
        self.assertGreater(adjusted.iloc[4], tg.iloc[4]["off_ppa_roll"])
        later = tg.copy()
        later.loc[later["game_id"] == "g2", "def_ppa_roll"] = 1000
        # An opponent's later rating does not rewrite A's pregame g2 feature.
        self.assertAlmostEqual(adjusted.iloc[4], _opponent_adjust(
            later, "off_ppa_roll", "def_ppa_roll").iloc[4])

    def test_elo_regresses_at_new_season(self):
        base = pd.DataFrame([
            ("2025-09-01", 2025, "A", "B", 40, 0),
            ("2026-09-01", 2026, "A", "B", np.nan, np.nan),
        ], columns=["date", "season", "home_team", "away_team", "home_points", "away_points"])
        base["date"] = pd.to_datetime(base["date"], utc=True)
        base["week"] = 1
        base["game_id"] = ["2025-1", "2026-1"]
        base["neutral_site"] = True
        p = _elo_probs(base)
        self.assertEqual(p.iloc[0], 0.5)
        self.assertGreater(p.iloc[1], 0.5)
        self.assertLess(p.iloc[1], 0.54)  # regression halves the first rating gain

    def test_rankings_only_apply_to_pregame_pair_and_are_capped(self):
        kickoff = pd.Timestamp("2026-10-03T17:00:00Z")
        published = pd.Timestamp("2026-09-27T12:00:00Z").to_pydatetime()
        scores = {"A": 0.9, "B": 0.3}
        self.assertAlmostEqual(adjustment("A", "B", kickoff, published, scores), 0.02)
        self.assertEqual(adjustment("A", "C", kickoff, published, scores), 0)
        self.assertEqual(adjustment("A", "B", kickoff, kickoff.to_pydatetime(), scores), 0)

    def test_current_form_replaces_last_season_by_game_eight(self):
        games, stats = [], []
        for i in range(12):
            season = 2025 if i < 3 else 2026
            week = i + 1 if i < 3 else i - 2
            gid = str(i)
            games.append((gid, season, week, pd.Timestamp("2025-08-01", tz="UTC")
                          + pd.Timedelta(days=7 * i), "A", "B", False,
                          28 if i < 11 else np.nan, 7 if i < 11 else np.nan))
            for team in ("A", "B"):
                val = 0.1 if season == 2025 else 0.9
                if i < 11:
                    stats.append((gid, team, val, 0.4, 1.0, 0.1, 0.4, 1.0))
        base = pd.DataFrame(games, columns=["game_id", "season", "week", "date",
                                            "home_team", "away_team", "neutral_site",
                                            "home_points", "away_points"])
        adv = pd.DataFrame(stats, columns=["game_id", "team", "off_ppa",
                                           "off_success", "off_explosive", "def_ppa",
                                           "def_success", "def_explosive"])
        tg = _team_game_long(base, adv)
        team = tg[tg.team.eq("A")].set_index("game_id")
        self.assertAlmostEqual(team.loc["3", "off_ppa_roll"], 0.1)
        self.assertGreater(team.loc["5", "off_ppa_roll"], 0.1)
        self.assertLess(team.loc["5", "off_ppa_roll"], 0.9)
        self.assertAlmostEqual(team.loc["11", "off_ppa_roll"], 0.9)


if __name__ == "__main__":
    unittest.main()
