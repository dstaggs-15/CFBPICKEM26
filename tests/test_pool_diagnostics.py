import unittest
from pool_diagnostics import report

class PoolDiagnosticsTests(unittest.TestCase):
    def test_reported_weeks_do_not_invent_favorite_comparisons(self):
        result = report({'season': 2026, 'weeks': [
            {'week':1,'wins':7,'losses':3,'source':'reported'},
            {'week':2,'wins':0,'losses':1,'picks':[
                {'pick':'A','winner':'B','market_favorite':'B'}]}]})
        self.assertEqual(result['games'],11)
        self.assertEqual(result['wins'],7)
        self.assertEqual(result['weeks'][0]['favorite_comparison_n'],0)
        self.assertEqual(result['weeks'][1]['disagreement_wins'],0)
        self.assertEqual(result['weeks'][1]['favorite_wins'],1)
