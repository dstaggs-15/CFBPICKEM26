# Upset watch research — September 2026

The new `upsets.html` page is a research view of **model-picked underdogs**.
It does not change the model's probability or pick. The betting line identifies
the underdog; ESPN's optional pick-share snapshot identifies a pick that few
pool entrants chose. The weekly pipeline regenerates `docs/upset_watch.json`
after `predict.py`. A crowd snapshot is used only when its season and week
match the predictions; otherwise the page still shows model-versus-favorite
disagreements without crowd numbers.

For each matchup, the page shows the current pregame offense-versus-defense
PPA comparison, pregame Elo and home field. It also counts prior-season FBS
underdogs in the same spread range, with the same home/away status and, when
there are at least 75 cases, the same direction of the PPA matchup advantage.
This is a descriptive sample, **not** an additional win probability or an
automated adjustment to the pick.

## Held-out results

On 7,002 FBS games with a nonzero saved line from 2017–2026, the model chose
1,262 line underdogs. Those teams won **39.4%** of their games. For line
underdogs of 0–3 points it was 48.4% (554 games); for 3–7 it was 39.2%
(446 games); for 7–14 it was 23.0% (196 games); and for 14+ it was 13.6%
(66 games). The model's average estimated probability on those 1,262 picks
was about 60.5%, so this disagreement subset was overconfident. The last
group has a small sample and 2026 is a partial season.

We evaluated two more detailed approaches without using any result from a
test season to create its predictors:

- A 75-neighbor historical profile method using opponent-adjusted pregame
  offense, defense, success, explosiveness, Elo, rest, venue and optionally
  spread. On 2020–2026 FBS games, its Brier error was 0.1776–0.1779,
  compared with 0.1756 for a simple line-only model fitted on earlier seasons.
- A regularized underdog model with those inputs plus four pregame passing and
  rushing offense-versus-defense matchups. On 2017–2026 FBS games, Brier
  error was 0.1724 for logistic regression and 0.1733 for a restricted
  boosting model, compared with 0.1718 for the line-only reference.

These experiments did not improve the held-out probability score. No
passing/rushing or analog signal was added to the production pick engine.
Finding an upset with a model probability over 50% is a prompt to investigate
the matchup, not evidence that it wins more than half of such disagreements.

## Crowd data

The initial Week 5 ESPN pick-share percentages were transcribed from the
owner-supplied Week 5 PDF into `docs/input/crowd_picks.json`. They are a
snapshot and can change before games lock. Future weeks need a new snapshot
if the page is to compare with the ESPN crowd; editing `games.txt` alone
does not contain pick-share data. Historical ESPN pick shares were not in the
repository, so this view has not been backtested against the actual pool
crowd. Pick share is not a bookmaker probability.

Run `python upset_watch.py` after `python predict.py`. The weekly GitHub
Actions pipeline does this automatically and publishes the JSON alongside
the normal predictions.

After `python fetch_cfbd.py --seasons 2014-2026` and `python features.py`, run
`python upset_analog_probe.py` to repeat the comparable-profile experiment.
It writes a diagnostic Parquet file under ignored `data/derived/` and does
not alter production picks. The passing/rushing experiment used the separate
research branch's advanced feature build; its [full fetch and feature-run
log](https://github.com/dstaggs-15/CFBPICKEM26/actions/runs/36517005972)
shows the source data and original candidate comparison.
