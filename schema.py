"""
schema.py — the single source of truth for what a 'game' row looks like.

Every other module imports from here. If a column name or expectation needs to
change, it changes in exactly one place. Last year's repo had column
assumptions scattered across a dozen files; that is how features silently
vanished. Not this time.
"""

from __future__ import annotations

# --- Identity columns: uniquely locate a game in space and time ---
ID_COLS = [
    "game_id",       # unique string id from the data source
    "season",        # int, e.g. 2025
    "week",          # int
    "date",          # tz-aware UTC timestamp
    "home_team",     # canonical team name
    "away_team",     # canonical team name
    "neutral_site",  # bool — critical: home-field logic must NOT apply when True
]

# --- Outcome columns: only known AFTER the game is played ---
# These may be null for future games we are predicting. They are NEVER features.
OUTCOME_COLS = [
    "home_points",
    "away_points",
]

# The label the model predicts. Derived from outcomes at training time only.
TARGET = "home_win"  # 1 if home_points > away_points else 0

# --- Feature groups ---
# Contextual features: known before kickoff, always available.
CONTEXT_FEATURES = [
    "is_postseason",     # bool/int
    "rest_diff",         # home_rest_days - away_rest_days
    "travel_diff_km",    # away travel distance proxy (home ~ 0)
]

# Team-strength features: opponent-adjusted, rolling, computed from PRIOR games only.
# These are the features that actually separate two evenly-matched teams —
# the ones that silently went missing last year.
STRENGTH_FEATURES = [
    "off_ppa_adj_diff",       # opponent-adjusted offensive PPA, home minus away
    "def_ppa_adj_diff",       # opponent-adjusted defensive PPA, home minus away
    "success_rate_adj_diff",  # opponent-adjusted success rate, home minus away
    "explosiveness_adj_diff", # opponent-adjusted explosiveness, home minus away
    "elo_home_prob",          # pregame Elo win prob for home team
]

# Market features: the BENCHMARK, deliberately kept out of the v1 model.
# We store them so the harness can score the model AGAINST the market,
# but the model does not get to peek at them.
MARKET_COLS = [
    "spread_home",       # points; negative = home favored (CFBD convention)
    "over_under",
    "market_home_prob",  # implied prob from the line
]

# The exact feature list the v1 model trains on.
MODEL_FEATURES = CONTEXT_FEATURES + STRENGTH_FEATURES

# Convenience: every column the canonical table should carry.
ALL_COLS = ID_COLS + OUTCOME_COLS + CONTEXT_FEATURES + STRENGTH_FEATURES + MARKET_COLS

# Expected dtypes for validation. Kept loose (family, not exact) on purpose.
DTYPE_FAMILIES = {
    "game_id": "object",
    "season": "integer",
    "week": "integer",
    "home_team": "object",
    "away_team": "object",
    "neutral_site": "boolean",
    "home_points": "numeric",
    "away_points": "numeric",
}
