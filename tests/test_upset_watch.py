import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from upset_watch import crowd_lookup, historical_group


class UpsetWatchTests(unittest.TestCase):
    def test_stale_crowd_snapshot_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "crowd.json"
            path.write_text(json.dumps({"season": 2026, "week": 5,
                "games": [{"away": "A", "home": "B", "away_picked_pct": 90,
                           "home_picked_pct": 10}]}))
            self.assertEqual(crowd_lookup({"season": 2026, "week": 6}, path), ({}, None))

    def test_historical_lookup_never_uses_current_season_results(self):
        rows = []
        for year, win in [(2024, True), (2025, False), (2026, False)]:
            rows.append({"season": year, "home_classification": "fbs",
                         "away_classification": "fbs", "spread_home": 5.0,
                         "home_points": 20 if win else 7,
                         "away_points": 7 if win else 20,
                         "home_off_vs_away_def_ppa": .2,
                         "away_off_vs_home_def_ppa": .1})
        result = historical_group(pd.DataFrame(rows), 2026, 5.0, True, True)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["dog_wins"], 1)


if __name__ == "__main__":
    unittest.main()
