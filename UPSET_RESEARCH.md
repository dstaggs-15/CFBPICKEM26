# When does an underdog deserve the pick?

This test is about winning outright, which is what the ESPN pool needs. A team
covering the spread doesn't count as a successful upset pick.

## What was tested

I rebuilt the joint stats-and-Elo model's forecasts season by season. Each model
trained on completed FBS games from older seasons only. Then I identified the
underdog from the historical spread and checked its final score.

The main comparison uses 2017–2025, excluding the current 2026 season. There
were 965 model-picked underdogs with an available, nonzero line. This is a
different scope from the earlier 39% figure, which included partial 2026 data.

I tested spread size, model confidence, current-season sample size, and whether
statistical contributions and Elo supported the underdog. Statistical support
means the four fitted joint-profile terms, added together and oriented toward
the underdog, were positive. Elo support means its pregame underdog win estimate
was above 50%. These are different measures; neither is a new forecast.

## What the patterns showed

| Model-picked underdogs | Wins / games | Outright win rate |
| --- | ---: | ---: |
| All | 382 / 965 | 39.6% |
| Underdog spread of 3 points or less | 240 / 508 | 47.2% |
| Spread above 3 through 7 | 110 / 302 | 36.4% |
| Spread above 7 through 14 | 30 / 126 | 23.8% |
| Spread above 14 | 2 / 29 | 6.9% |
| Model probability 50% to under 55% | 155 / 413 | 37.5% |
| Model probability 55% to under 60% | 108 / 271 | 39.9% |
| Model probability 60% or more | 119 / 281 | 42.3% |
| Both stats and Elo supported the underdog | 80 / 189 | 42.3% |
| At least five earlier FBS games per team | 183 / 403 | 45.4% |

Small underdogs were a better place to look. More current-season games helped
in this descriptive comparison. Simply raising the model-confidence threshold
didn't make its upset forecasts reliable. Even the 60%+ group lost more often
than it won: the model's probabilities are overoptimistic in this subgroup.
Overall probability performance does not guarantee good calibration for upset
picks.

## Choosing a screen without using the test season's answers

The fixed search grid was:

- Model probability at least 50%, 55%, or 60%.
- Underdog spread at most 3, 7, 14 points, or unrestricted.
- No support requirement, positive statistical support, or stats and Elo support.
- At least zero, three, or five earlier FBS games for each team.

There are 108 candidate screens. A screen needed at least 100 earlier games
across three seasons to be considered. Selection maximized the lower end of an
approximate 95% Wilson interval, which discourages rewarding a tiny lucky group.
That interval alone does not correct for searching many rules.

To test the selection procedure, I chose a screen using only earlier held-out
forecasts, then applied it to the next season. This ran for 2020–2025. The
selected screens changed as more earlier seasons became available.

**That later-season test produced 127 wins in 272 picks: 46.7%.** Its approximate
95% interval was 40.8%–52.6%. Picking the favorites in those same games would
have produced 145 wins, or 53.3%. This does not establish an advantage for picking
the selected underdogs.

For 2026, the screen selected using 2017–2025 is:

1. Model probability at least 55%.
2. Underdog spread of 3 points or less.
3. Both teams have at least five earlier FBS games.
4. No extra stats/Elo agreement requirement.

Its selection sample was 81 wins in 149 games, or 54.4%, with an approximate
95% interval of 46.4%–62.2%. **That is a research result from the data used to
choose the screen, not independent validation.** It must not be sold as a proven
54% upset strategy. The 46.7% number above tests the changing selection procedure,
not this final rule alone across all six seasons.

## What changed on the website

Upset Watch now shows whether a model-picked underdog meets the screen and why
it fails. Passing an unvalidated screen receives a research label, not a strong
recommendation. A supported label requires at least 100 later-season test picks
and a lower approximate 95% confidence bound above 50%. This audit didn't pass
that requirement, so there are currently no supported upset recommendations.

Missouri's Week 5 pick is watch-only. Its 51.4% model estimate is below 55%, its
5.5-point underdog line is above 3, and the statistical snapshot has only three
prior FBS games per team. The main forecast still picks Missouri; this screen
doesn't silently replace it with Florida.

The screen is frozen by season and model version. The weekly pipeline uses the
existing screen; if a new season has no screen, it builds one from earlier
seasons. A missing or mismatched screen never produces a supported label.
Earlier picks, records and the separate rankings site are unchanged.
The first published screen for each week is also frozen under
`historicals/upset_watch/<model-version>/` so future assessments can be checked
against actual outcomes rather than reconstructed afterward.

## Limits and next evidence

These seasons have already been used during model development. Although each
forecast and each screen selection respects time ordering, this is retrospective
research, not an untouched final test. Game results can be correlated within
teams and seasons, so Wilson intervals are approximate. A later-season win rate
above 50% would be encouraging, but prospective frozen picks remain necessary.

The useful conclusion is narrower than "we solved upsets": small underdogs with
more statistical history deserve closer examination, but this model has not
shown when overriding the favorite improves outright winner accuracy. Don't
force an upset just because the model says 51%.

## Reproduce

After the normal feature build:

```sh
python upset_audit.py
python upset_watch.py
python -m unittest discover -s tests -v
```

The audit writes dated season policy and group summaries under
`historicals/model_audits/`, and held-out game-level evidence to
`data/derived/upset_evidence.parquet`. `--if-missing` preserves an existing
season policy. The four new tests cover future-outcome exclusion, minimum sample
size, labels for unvalidated screens, and preservation of the original pick.
All 27 project tests passed.
