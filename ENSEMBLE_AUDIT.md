# Multi-view challenger research — September 2026

`ensemble_challenger.py` builds one coherent research challenger from five
pregame views: the existing opponent-adjusted boosted win model, a separate
offense-versus-defense matchup classifier, a projected scoring margin, pregame
Elo, and the saved spread probability. A regularized logistic combination is
fitted on **earlier seasons' out-of-sample base predictions** and evaluated on
the following season. It also evaluates the four-view independent version and
a simple 25% model / 75% spread blend. All base learners are fitted only on
seasons before the game being forecast. Matchup and margin models use the
production feature list; no passing/rushing research columns are required.

This is a diagnostic, not a replacement for the live independent model. The
weekly pipeline does not call this script, so picks and displayed confidence
remain unchanged.

| 2020–2025 FBS games with a nonzero saved spread (4,471) | Correct | Accuracy | Brier ↓ |
| --- | ---: | ---: | ---: |
| Existing independent model | 3,129 | 69.98% | 0.19995 |
| Independent four-view combination | 3,134 | 70.10% | 0.19375 |
| Saved spread probability | 3,254 | 72.78% | 0.17919 |
| Spread probability refitted on prior out-of-sample seasons | 3,255 | 72.80% | **0.17778** |
| 25% model / 75% spread | **3,260** | **72.91%** | 0.17895 |
| Full five-view combination | 3,252 | 72.74% | 0.17801 |

The independent combination improves probability error, but barely changes
winner accuracy and still trails the saved favorite. The full combination
does not beat a spread-only refit in Brier or accuracy. The fixed 25% blend
gets six more games correct than the saved-spread benchmark across six seasons,
but has higher Brier than the prior-season spread refit and is not a credible
75% solution. In 2023–2025, it gets 13 more right than the raw spread, but
its Brier is slightly worse. These candidate comparisons are exploratory;
multiple ideas were tried. 2026 is partial and not included in the table.

The archived CFBD spread may reflect a different time from a real weekly pick
deadline; retrospective comparisons cannot establish what the blend would
have known then. We need timestamped line and roster snapshots to evaluate
a real pre-pick-time hybrid. Historical ESPN pool slates and crowd shares are
also incomplete, so these numbers describe all lined FBS games, not the
owner's ten selected games per week.

To reproduce after `features.py` has built the training Parquet:

```sh
python ensemble_challenger.py
```

The optional `--heldout-cache` accepts a Parquet containing a `game_id` and
an out-of-season `p_model` column, allowing quick reruns. The default rebuilds
the forecasts with `audit_fbs.heldout()`. `--output data/derived/ensemble_oos.parquet`
saves detailed research forecasts outside the tracked site files.

The next useful data experiment is a historically timestamped quarterback and
roster availability signal plus prior-game drive/field-position measures. Test
each on later seasons before assigning it a live weight.
