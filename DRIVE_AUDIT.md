# Pregame drive challenger — September 2026

CFBD provides historical drives with offense, defense, starting yards-to-goal,
and start/end scores. `drive_challenger.py` fetches them for 2014–2026, removes
overtime drives and duplicate IDs, and summarizes points per drive and starting
field position for each team/game. Its matchup inputs use only *earlier
completed games*: last-eight-game form in the current season with a fading
previous-season prior. The game being predicted cannot enter its own inputs.
No betting-line feature enters either version of the classifier.

The script fits the unchanged production feature list and the same model plus
three drive features on earlier seasons, then compares each on the next
season's FBS-vs-FBS games. It is research-only; the weekly model, picks and
site confidence are unchanged.

| 2017–2026 held-out FBS games (7,015) | Accuracy | Brier ↓ |
| --- | ---: | ---: |
| Existing model, rebuilt in the same run | **69.71%** | 0.19823 |
| With prior-game drive matchup and field position | 69.44% | **0.19762** |
| Saved betting spread reference | 73.66% | 0.17307 |

The new view improved probability error by 0.00061 but reduced correct
winners. In 2023–2025 alone the drive version was better in 2025, worse in
2023 and 2024. This does not support putting it into the live pick engine.
Drive feature coverage was 99.9% in the 7,015 FBS test games. The raw drive
count changes sharply in 2022; any future use should verify provider coverage
and consistency by season rather than assuming identical collection.

The full [GitHub Actions run](https://github.com/dstaggs-15/CFBPICKEM26/actions/runs/36580904750)
fetched games and drives with the repository's existing CFBD key and uploaded
the per-season audit. Eight unit tests passed locally, including checks that
the current game's drives cannot enter its pregame form.

The CFBD `/player/returning` endpoint returns team/season returning-production
metrics without a historical observation timestamp. Its transfer endpoint
has a transfer date, but the currently returned destination is not proof of
when that destination became known. Neither provides a historical, dated
quarterback injury/availability series. We cannot backfill such data as if it
were known before past kickoffs. A future availability feature needs archived
pregame snapshots with team, player, status, source and captured-at time, plus
enough weeks of outcomes to validate it on later games.

Reproduce after the normal historical feature build:

```sh
python drive_challenger.py --fetch --seasons 2014-2026
```

The research data and diagnostic Parquet stay under ignored `data/`; the
workflow has read-only repository permissions and publishes no predictions.
