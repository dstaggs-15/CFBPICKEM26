"""Production winner policy validated against saved FBS lines.

The independent classifier remains market-free. This final probability blend
combines its estimate with the repository's spread-derived reference.
"""
from math import isfinite

MODEL_WEIGHT = 0.25
POLICY = "football25_spread75_v1"


def combine(independent_probability, market_probability):
    independent = float(independent_probability)
    if not isfinite(independent) or not 0 <= independent <= 1:
        raise ValueError("Independent probability must be finite and between 0 and 1")
    if market_probability is None:
        return independent, 1.0
    market = float(market_probability)
    if not isfinite(market):
        return independent, 1.0
    if not 0 <= market <= 1:
        raise ValueError("Market probability must be between 0 and 1")
    return MODEL_WEIGHT * independent + (1 - MODEL_WEIGHT) * market, MODEL_WEIGHT
