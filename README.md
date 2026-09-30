# CFB Pick'em Model

This project picks the winners of the ten games in my weekly ESPN college football pool. The site shows the pick, estimated win probability, team stats and the reasons behind it.

[View the picks](https://dstaggs-15.github.io/CFBPICKEM26/) · [Season record](https://dstaggs-15.github.io/CFBPICKEM26/record.html) · [Separate rankings](https://dstaggs-15.github.io/cfbranking/)

## How a pick gets made

The model learns from past FBS games. For each historical game, its inputs describe what was known before kickoff. The final score tells it which side won. It learns how strongly each input tends to be associated with winning, then applies those weights to the current matchup.

The current predictor is regularized logistic regression. In plain English, it is a learned scorecard: each input adds or subtracts from a score, and that score becomes a win probability. Regularization keeps the fitted weights from growing too large just to explain a few unusual games.

It does not have a rule that says Alabama gets extra points because it is Alabama. Team names and betting lines are not prediction inputs. It does use Elo, a rating updated from earlier wins and losses. That is still a history-based signal, and it currently has the largest fitted weight. Each offseason, rating gaps are cut in half toward an average team.

The model picks the home team at 50% or higher; otherwise it picks the away team. The separate rankings can make a very small adjustment afterward.

## What it looks at

| Input | What it tells the model |
| --- | --- |
| Offensive efficiency | How productive each offense has been per play, adjusted for opponents faced |
| Defensive efficiency | How much each defense allows per play, adjusted for opponents faced; lower is better |
| Success rate | How consistently an offense produces successful plays |
| Explosiveness | How productive its successful plays are |
| Offense/defense matchup terms | Comparisons between each offense and the defense it is about to face |
| Elo | A pregame strength estimate updated from wins and losses, with an offseason reset toward average |
| Venue | Whether the game is at a home stadium or a neutral site |
| Rest | The difference in days since each team's previous scheduled game |
| Postseason | Whether this is a postseason game |

PPA is the efficiency measure used here. It estimates how much a play changes expected scoring value. It helps distinguish productive play from simply running up yardage.

## The weights

There is no fixed recipe like “40% offense, 30% defense, 30% history.” The model learns its weights when it trains. They can change on the next run.

These are the fitted weights from the September 30, 2026 model. Inputs are first put on a common scale. A weight describes the change in the model's score for a one-standard-deviation increase in that input. **These are not percentage-point changes in win probability.**

| Input | Fitted weight | Size relative to Elo |
| --- | ---: | ---: |
| Home team's Elo win estimate | +0.7583 | 1.00 |
| Home minus away success rate | +0.2734 | 0.36 |
| Home minus away defensive PPA allowed | −0.2641 | 0.35 |
| Home minus away offensive PPA | +0.1846 | 0.24 |
| Neutral-site indicator | +0.0937 | 0.12 |
| Away offense minus home defense PPA | +0.0436 | 0.06 |
| Home minus away explosiveness | +0.0226 | 0.03 |
| Home minus away rest days | −0.0189 | 0.02 |
| Postseason indicator | +0.0115 | 0.02 |
| Home offense minus away defense PPA | −0.0105 | 0.01 |

A positive term pushes the score toward the home team; a negative term pushes it toward the away team. The defense weight is negative because allowing less PPA is better. These inputs overlap, so a coefficient should be read alongside the others. For example, the rest coefficient does not prove that extra rest hurts a team.

The score also includes a starting value of 0.4055 and small terms for missing efficiency inputs. Missing values are filled using training-data averages, with flags that tell the model the original value was unavailable. Those flags each have a fitted weight of about +0.0033 in this snapshot.

The “relative to Elo” column compares coefficient sizes, not shares of the final pick. It does not add up to 100%. A large weight contributes little when the teams are close on that input; a smaller weight can matter when the gap is large.

For anyone checking the math: the model sums the starting value and all weighted, scaled inputs into a score `z`, then calculates `P(home win) = 1 / (1 + exp(-z))`. The card's supporting inputs come from the actual fitted terms for that game.

## How much of last season carries over?

Efficiency uses up to eight observed games from the current season. Last season's final eight observed games provide a small starting prior that fades as new stats arrive.

| Observed current-season games | Last season's remaining weight | Last season's share of the efficiency average |
| --- | ---: | ---: |
| 3 | 1.25 games | 29.4% |
| 4 | 1 game | 20.0% |
| 8 or more | 0 games | 0% |

Before both teams have three observed games, the six efficiency comparison inputs are left missing on purpose. Elo, venue and the other context inputs still work. The table above describes the rolling efficiency average, not the share of the entire prediction.

Future scheduled games do not count as observations. They cannot push real games out of the window. The opponent-strength window also resets each season.

## The similar historical games on each card

This is the “these teams look like past teams” part.

The site compares both teams' pregame offense, opposing defense, success rate and explosiveness with earlier-season FBS matchups at the same home/neutral venue type. It finds 75 similar profiles, shows how often the matching side won, and lists the three closest games with their final scores.

Team names, Elo, betting lines and game results are not used to decide which profiles are similar. Results are checked after the similar games have been selected.

That historical win rate is context, not a second prediction added to the model. We tested a nearest-neighbor predictor, and it performed worse than the chosen statistical model. A card can therefore show a model pick that disagrees with its historical comparison sample. That disagreement is useful to see rather than hide.

## What changed on September 30

- Replaced the boosted-tree predictor with the tested logistic model, trained on completed FBS-versus-FBS games.
- Fixed rolling windows so future scheduled games cannot erase observed form.
- Reset opponent-strength history each season and made history counts reflect the current season.
- Added historical matchup examples and explanations based on the fitted model's actual terms.
- Kept early-season missing inputs separate from a broken data feed. Mature profiles must pass the coverage checks.
- Reduced the separate rankings adjustment from a maximum of two percentage points to **0.05 percentage points**. The rankings site itself was not changed.
- Added a separate frozen archive and record for the updated model's picks.

## What the testing showed

Each test season was predicted using a model trained only on earlier seasons. On the same 6,800 FBS games from 2017–2025:

| Model | Winner accuracy | Brier score, lower is better |
| --- | ---: | ---: |
| Previous boosted model | 69.60% | 0.19878 |
| Updated statistical model | 70.44% | 0.19370 |

Brier score measures how far the probabilities were from the results, including how costly confident mistakes were. Both overall measures improved. The most recent three seasons were nearly tied in winner accuracy: 70.48% before and 70.60% after.

The original pool record is 21–19 through four weeks. The historical replay of the twenty archived, completed pool games did not show an accuracy improvement. These tests do not establish a 73–75% hit rate for the ten-game pool. The updated model's future frozen picks will give us that report card.

The [research notes](PROFILE_RESEARCH.md) cover the other models tested, repositories reviewed and the limits of these comparisons.

## The separate rankings and betting line

The ranking numbers beside teams come from the separate computer-ranking site, not an official poll. A prediction can use its scores only when both teams appear in a valid, recent snapshot published before kickoff. Missing teams are not assigned an invented rank.

The adjustment is `0.04 × (home score − away score)`, capped at `±0.0005` probability, or **±0.05 percentage points**. A 60.00% estimate can move at most to 60.05% or 59.95%. Its predictive value has not been established by a historical test.

Betting lines are used for comparison and the upset-watch page. They do not choose the model's winner. News headlines appear on the cards but do not change the probability. Injuries and roster news are not yet trained prediction inputs.

## Updating a week

1. Edit `docs/input/games.txt` with the ten ESPN matchups, one per line. Use `Away @ Home` or `Team A vs Team B` for a neutral-site game.
2. In GitHub Actions, open **Run weekly pipeline** and select **Run workflow**. The normal season range is `2014-2026`.
3. The job runs the tests, fetches CFBD data, grades completed archives, builds features, trains and tests the model, then publishes the picks and news.

The CFBD key stays in the repository's `CFBD_API_KEY` secret. Predictions are refused after kickoff. A rerun can refresh the board before kickoff, but it does not replace a version's first archived picks for grading.

The original weekly archives live in `historicals/predictions/`. Updated-model archives live in `historicals/model_versions/statistical-logistic-v2/`. The record page shows both records; results are graded once every game in the archived slate has a final score. Weeks 1–2 have reported totals without individual archived picks.

Optional ESPN crowd shares go in `docs/input/crowd_picks.json`. A stale snapshot is ignored. **Refresh news** can run separately without rebuilding predictions.

## Running locally

```sh
pip install -r requirements.txt
python -m unittest discover -s tests -v
python fetch_cfbd.py --seasons 2014-2026
python grade_results.py
python pool_diagnostics.py
python features.py
python train_and_backtest.py
python predict.py
python upset_watch.py
```

Set `CFBD_API_KEY` in your environment before fetching. The audit commands and results are in [PROFILE_RESEARCH.md](PROFILE_RESEARCH.md). The drive and ensemble experiments remain research-only; they do not feed the live picks.
