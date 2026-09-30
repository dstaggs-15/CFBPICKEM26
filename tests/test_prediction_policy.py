import unittest
from prediction_policy import combine


class PredictionPolicyTests(unittest.TestCase):
    def test_market_can_correct_overconfident_underdog_pick(self):
        probability, weight = combine(.7, .3)
        self.assertAlmostEqual(probability, .4)
        self.assertEqual(weight, .25)

    def test_missing_market_preserves_independent_forecast(self):
        for missing in [None, float('nan')]:
            self.assertEqual(combine(.7, missing), (.7, 1.0))

    def test_invalid_probabilities_are_rejected(self):
        for independent, market in [(float('nan'), .5), (.5, 1.5)]:
            with self.assertRaises(ValueError):
                combine(independent, market)
