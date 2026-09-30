# Testing whether current stats should outweigh Elo

The historical examples on the cards do not drive the live prediction. That
doesn't fully deliver the original request for a model built around comparable
statistical profiles. Historical game outcomes train the live model, but that
is different from using the displayed 75-game sample as its prediction.

## A stronger stats experiment

`joint_profile_audit.py` estimates offense, defense, and home field together.
Instead of judging an offense against an opponent's simple average, it solves
the whole current-season schedule as one set of equations. It does this for
PPA, success rate, explosiveness, and points scored. Ridge regression shrinks
thin samples toward the current-season average.

For each upcoming matchup, it calculates the expected difference between the
home offense against the away defense and the away offense against the home
defense. A separate logistic model learns how those statistical differences
related to wins in older seasons. It also considers rest, neutral sites, and
postseason games. There is no Alabama reputation bonus, previous-season team
rating, ranking score, or betting line in the stats-only candidate.

Team names connect a team's games inside a season's schedule equations. They
are never inputs to the historical winner model. Ratings restart each season.
Profiles use completed FBS games before that week's Monday, and need at least
two earlier FBS games for each team. Missing early-season profiles are imputed
inside the training pipeline, with missingness indicators.

I tested three shrinkage settings, a stats-only winner model, the same model
with Elo, and a distance-weighted 75-neighbor classifier. The neighbor classifier
actually uses historical outcomes to forecast; it is not a display feature.

## Results

These are straight-up winners, not spread picks. Each tested season's winner
model is trained only on older seasons. On 6,800 FBS games from 2017–2025:

| Model | Winner accuracy | Brier error (lower is better) |
| --- | ---: | ---: |
| Current live model, season-held-out replay | 70.44% | 0.19370 |
| Joint current-season stats, shrinkage 1 | 68.68% | 0.19899 |
| Joint stats plus Elo, shrinkage 1 | 70.31% | 0.19103 |
| 75 similar profiles, shrinkage 1 | 68.00% | 0.20411 |

Selecting a setting because it won this whole table would overstate the result.
The script also chooses each season's shrinkage and blend using only earlier
held-out seasons. That comparison covers 4,478 games from 2020–2025:

| Model | Winner accuracy | Brier error |
| --- | ---: | ---: |
| Current model | 69.72% | 0.19664 |
| Stats only, selected on older seasons | 68.31% | 0.20235 |
| Stats plus Elo, selected on older seasons | 69.67% | 0.19364 |
| Current model blended with stats-only candidate | 69.61% | 0.19488 |

The blend selected 40% stats in 2020, then 30% in 2021–2026. Those percentages
came from earlier predictions and outcomes, not a manually assigned ratio.
They improved probability error but did not improve winner accuracy.

These remain exploratory retrospective tests. We have already examined these
seasons during model development. Time ordering prevents future outcomes from
entering individual forecasts; it does not turn reused research data into an
untouched final test. Nor are these results a 73–75% ESPN pool record. The pool
slate is a smaller and different sample.

## What this says about the two disputed picks

Using only pre-2026 games to train the winner model, and current-season profiles
before September 28 for the scheduled Week 5 games:

| Home team vs away team | Stats-only home win chance | Stats plus Elo home win chance |
| --- | ---: | ---: |
| Mississippi State vs Alabama | 54.72% | 43.63% |
| Missouri vs Florida | 41.93% | 51.41% |

Elo reverses both picks in this experiment. The concern about its influence is
real. These challenger probabilities are research output, not replacement
published picks. The displayed 37% Alabama comparison sample uses a different
distance definition and should not be described as the challenger forecast.

## Decision

Keep the live predictions unchanged for now. Removing Elo or promoting the
neighbor vote would reduce historical winner accuracy in these tests. The joint
stats plus Elo candidate is worth tracking prospectively because its probability
error improved, but it hasn't demonstrated the requested gain in correct picks.

Use joint opponent-adjusted current-season profiles as the next research basis.
Test richer play-level matchup inputs and sample uncertainty before choosing
new weights. Compare each candidate with Elo alone, the live model, and market
favorites on the actual pool. Freeze each forecast before kickoff. Promote a
candidate when it improves the agreed evaluation, rather than when it produces
more appealing explanations. Do not force agreement with 75 neighbors or invent
a fixed historical-statistics percentage.

## Reproduce

With the existing training and advanced-stat parquet files:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python joint_profile_audit.py
python -m unittest discover -s tests -v
```

Outputs go under `data/derived/joint_*`. The committed metric and selection
snapshots are under `historicals/model_audits`. Two new tests check that future
results cannot alter a pregame profile and that names or a previous season
cannot create a reputation carryover. All 20 project tests passed.

Sources:
- [CFBD opponent-adjusted statistics using simultaneous ridge regression](https://radsportsanalytics.com/blog/opponent-adjusted-stats-ridge-regression/)
- [Original opponent-adjustment code](https://github.com/jbuddavis/opponentAdjustedStats/blob/main/oppAdjPBP.py)
- [scikit-learn guidance on time-ordered validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Why model selection needs separate evaluation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

This experiment adapts the ridge method to aggregate game observations; it does
not reproduce the source's play-level model or reuse its penalty values.
