# Current team profiles and historical matchups

The goal is to learn what happens when current pregame profiles meet, rather
than awarding a team points for its name or an old championship. Production now uses regularized logistic regression with opponent-adjusted
efficiency and pregame Elo; the previous predictor used boosted trees. Elo still carries a regressed prior-season rating; it is not team identity
encoded as a categorical input. The previous-season efficiency prior starts at
two games of weight and reaches zero after eight observed current-season games.

## Research references

- [TDNet](https://github.com/T-Burton-ND/TDNet): time-dependent team fingerprints,
  several statistical learners including nearest neighbors, frozen predictions
  and cutoff audits. Its design does not establish that our model improves.
- [CFB-Model](https://github.com/blaizerlahman/CFB-Model): lagged team/opponent
  rolling statistics and historical weekly rating snapshots. Its main task is
  spread prediction, which differs from our straight-up pool winner task.
- [CFB model data sheet](https://github.com/zachringnight/cfbmodel/blob/main/info_sheet_data.md):
  opponent-adjusted feature rows using only games preceding the matchup.

## Changes

Only completed games with observed statistics occupy rolling windows. Future
scheduled games no longer evict observed form. History counts are current-season
counts. Opponent schedules and their reference means reset each season rather
than applying old opponents to current-season performance.

`profile_analogs.py` compares eight pregame efficiency measures across both
teams: offense, opposing defense, success rate and explosiveness. It standardizes
using earlier-season completed FBS profiles only and retrieves 75 neighbors with
the same neutral/home venue context. Neither team names, results, betting lines
nor Elo enter the distance. The board shows three nearest examples and the
observed outcome count. That rate is descriptive and does not override the
calibrated classifier. Differences in era, roster and unobserved circumstances
mean a neighbor is not an exact match.

The rankings site is read only. Its unvalidated live adjustment is capped at
0.05 percentage points instead of two percentage points. Its historical effect
is not included in model accuracy comparisons.

`profile_audit.py` compares the frozen original builder, corrected builder,
regularized logistic regression and 75-neighbor classification. Each test season
uses models trained only on earlier seasons; 2026 is partial and diagnostic.
It reports full-FBS, early-season and recent-season results separately, plus a
replay subset defined by archived pool slates. Replay predictions are not the
original published weekly probabilities. No 73–75% claim follows from an
exploratory candidate comparison.

`pool_diagnostics.py` summarizes actual published pool results and disagreement
performance. Owner-reported weeks count toward the overall record but do not
invent a favorite comparison where individual picks are missing.

## Reproduce

```sh
python -m unittest discover -s tests -v
python fetch_cfbd.py --seasons 2014-2026
mkdir -p data
git show 722046606b7214fc6630068158203f9ccc430db2:features.py > data/original_features.py
python profile_audit.py
python pool_diagnostics.py
```

The audit workflow saves the metric JSON, out-of-season predictions and fetched
data as artifacts. The weekly pipeline runs the invariant tests before fetching
and forecasts only games that have not kicked off. Existing grading archives
remain immutable. The live board can be refreshed before kickoff; its original
weekly archive is still the grading source.

## Results and production decision

The [audit run](https://github.com/dstaggs-15/CFBPICKEM26/actions/runs/36732456326)
completed successfully. On the same 6,800 FBS games from 2017–2025:

| Candidate | Winner accuracy | Brier (lower is better) |
| --- | ---: | ---: |
| Original boosted model | 69.60% | 0.19878 |
| Boosted model with corrected profiles | 69.04% | 0.20127 |
| Corrected profiles + regularized logistic model | **70.44%** | **0.19370** |
| Corrected profiles + nearest-neighbor predictor | 69.06% | 0.19679 |
| Direct efficiency profiles, logistic, excluding Elo | 68.41% | 0.20344 |
| Direct efficiency profiles, boosted, excluding Elo | 66.13% | 0.21359 |

The last two variants were tested locally on the exact downloaded audit data
using `efficiency_audit.py`. These are exploratory comparisons; the same seasons
have been inspected for several candidates, so they are not pristine final
confirmatory test sets. The production decision favors the regularized logistic
model for its overall accuracy, probability error and exact interpretable
feature contributions. In 2023–2025 its accuracy was 70.60% versus 70.48% for the
old model: only three more correct games out of 2,398. Its Brier was 0.19471
versus 0.19662. No large recent-season accuracy improvement is established.

2026 partial-season full-FBS logistic accuracy was 74.88% on 215 games, but its
Brier was worse than the old model on that subset. This is diagnostic and is
not a 75% pool accuracy claim. The initial workflow replay resolved ten archived games by ID. Week 3's legacy
archive omitted IDs; resolving unique season/week/team matches expands the
replay to twenty completed games. Both the original and new logistic model
went 12–8 in this season-held-out replay; logistic Brier was worse (0.25808 vs
0.24694). This is not the original published 9–11 result for those weeks:
season-held-out models did not retrain within 2026. Weeks 1–2 lack per-game
archives. The twenty-game subset is too small to establish a pool improvement;
the published results remain the source for the original 21–19 pool record.

`StatisticalModel` trains exclusively on completed FBS-vs-FBS games. Its
imputation, standardization and regularized feature weights are fitted on the
training data. It uses no market feature or team identity. The board's strongest
supporting inputs now come from exact additive fitted log-odds contributions;
those are saved in prediction JSON for review. Coefficients express statistical
association, not causal effects. Historical neighbors remain descriptive because
the predictive neighbor and pure-profile variants did not improve performance.

Run `python efficiency_audit.py` after `profile_audit.py` to repeat the additional
no-Elo candidates. Accuracy on new frozen pool picks remains the real target.

The updated version's first Week 5 predictions are frozen separately under
`historicals/model_versions/statistical-logistic-v2/`. The weekly grader produces
`docs/model_version_results.json`; the record page reports these prospective
results alongside the original record. No previous archive is overwritten.
The coverage contract distinguishes deliberately gated early games from eligible
mature profiles: at least 95% efficiency coverage is required on eligible rows,
with per-season feed-outage checks. All FBS rows retain context and Elo checks.
