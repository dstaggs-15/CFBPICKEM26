import json
import tempfile
import unittest
from pathlib import Path
from datetime import date
from predict import save_archive


class ArchivePolicyTests(unittest.TestCase):
    def test_pre_game_policy_change_preserves_original_and_reissues(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = dict(season=2026,week=5,prediction_policy='old',games=[
                dict(game_id='1',game_date='2026-10-03',pick='A')])
            new = dict(old,prediction_policy='new',games=[
                dict(game_id='1',game_date='2026-10-03',pick='B')])
            save_archive(old,root,date(2026,9,30))
            save_archive(new,root,date(2026,9,30))
            self.assertEqual(json.loads((root/'predictions/2026-week-5.json').read_text()),new)
            self.assertEqual(json.loads(next((root/'revisions').glob('*.json')).read_text()),old)

    def test_game_day_or_same_policy_cannot_replace_recorded_picks(self):
        for today,policy in [(date(2026,10,3),'new'),(date(2026,9,30),'old')]:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                old=dict(season=2026,week=5,prediction_policy='old',games=[
                    dict(game_id='1',game_date='2026-10-03',pick='A')])
                save_archive(old,root,date(2026,9,29))
                new=dict(old,prediction_policy=policy,games=[
                    dict(game_id='1',game_date='2026-10-03',pick='B')])
                save_archive(new,root,today)
                self.assertEqual(json.loads((root/'predictions/2026-week-5.json').read_text()),old)
