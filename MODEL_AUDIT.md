# FBS model audit — September 2026

This audit scores games from 2017–2026. Every fold trains on earlier seasons
and evaluates the next season. It includes 7,015 completed FBS-vs-FBS games;
2026 is partial. The classifier trains on the repository's full historical
game table, but all numbers below are scored on identical FBS games. Betting
lines are held out of the classifier and used only as a benchmark.

## What changed

The probability calibrator changes from isotonic to a smoother sigmoid fit on
out-of-sample predictions for the most recent available prior season. The
final classifier still trains on all earlier seasons, so this calibration is
an approximation; the outer season-by-season evaluation tests that complete
procedure on unseen seasons.

| Version | FBS winner accuracy | Brier (lower is better) | Log loss |
| --- | ---: | ---: | ---: |
| Existing isotonic | 69.61% | 0.1992 | 0.5955 |
| Sigmoid | 69.71% | 0.1982 | 0.5808 |

Differences in winner accuracy are small; these numbers do not establish that
the new model beats the betting favorite. The saved favorite won 73.66% on
those same games. See the [full experiment run](https://github.com/dstaggs-15/CFBPICKEM26/actions/runs/36517005972).

## Where it struggles

With the new sigmoid calibration, local reruns gave 68.1% in weeks 1–4,
71.6% in weeks 10+, and 58.6% on neutral sites. The saved favorite won
67.3% of the 539 neutral-site games. In the 1,270 games where the new model
and saved favorite chose different winners, the model won 39.2%. These are
diagnostics for investigation, not rules to override a future pick.

Games assigned at least 80% confidence won 85.6% of the time across 1,890
games (26.9% of the FBS evaluation). Individual seasons ranged from 81.1%
to 90.6%, so this is not a guaranteed weekly hit rate. The game pool on the
site is also a selected slate, not the full FBS schedule.

## Candidates rejected for the live model

The existing game-stat endpoint includes separate passing and rushing PPA.
Adding four offense-versus-defense passing/rushing terms provided 98.1% FBS
coverage, but accuracy was 69.58% with isotonic versus 69.61% without them.
With sigmoid, the extra terms reached 69.61% versus 69.71% without them.
They did not improve early-week or neutral-site results either.

Changing the prior-season efficiency weight from two games to one reduced
opening-week accuracy; raising it to four improved early-week Brier slightly
but reduced overall accuracy. Changing Elo offseason regression from 0.50
to 0.35 helped early weeks but reduced overall winner accuracy. The existing
weight and regression remain in production.

Returning-production and transfer data could improve the preseason estimate.
The available season-level feed does not provide historical as-of dates for
each value. We need archived preseason snapshots before claiming a leak-free
historical test or using later roster information as if it were known at kickoff.

## Rerun

With `CFBD_API_KEY` set and dependencies installed:

```sh
python fetch_cfbd.py --seasons 2014-2026
python features.py
python audit_fbs.py
python train_and_backtest.py
```

The separate rankings-site adjustment occurs after the trained model's
prediction; historical ranking snapshots are not available for most test
seasons, so it is not included in these numbers.
