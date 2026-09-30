# Winner policy release — September 2026

The final site probability now combines 25% independent football estimate
with 75% of the existing spread-derived probability. The classifier still
learns solely from pregame football/context features. Its ranking-site nudge
is applied before blending, giving the ranking model at most 0.5 percentage
points of final influence when a spread is available. Missing spreads use the
independent forecast. The JSON records each component and the policy version,
and the pick explanation identifies the spread contribution.

This is an accuracy-oriented release, not a claim to beat the market. In the
completed 2021–2025 experiment (3,937 FBS games with nonzero saved lines), the
fixed blend got 2,869 winners right (72.87%); independent predictions got
2,752 (69.90%); the saved favorite got 2,860 (72.64%). Nine extra correct games
versus the favorite is a small exploratory difference, not an established
edge. Historical ranking snapshots were unavailable and excluded from the
test. Historical saved lines may differ from what was available at a weekly
pick deadline. The ESPN pool selects its own ten games, so its expected hit
rate is not established by an all-FBS average.

`policy_audit.py` rebuilds prior-season-only forecasts, measures the exact
blend on the same FBS games, and reports early-season, neutral-site, small-line
and underdog-disagreement results. The release workflow requires improved
winner accuracy over the current independent model on completed 2021–2025
seasons. Its report is retained as the `live-policy-audit` Action artifact.

The drive and multi-view candidates remain research-only because their tests
did not support a live accuracy improvement. Historical quarterback injury
availability has not been added without dated pregame records.

When a policy changes before *all* archived games' dates, `predict.py` preserves
the old first slate under `historicals/revisions/` and reissues the canonical
week archive using the new policy. After game day starts, or under the same
policy, that canonical archive remains immutable. This keeps the Week 5 record
aligned with the newly published picks without deleting earlier predictions.
