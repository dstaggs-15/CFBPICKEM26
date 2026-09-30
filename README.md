# CFB Pick'em Model

This project picks the winners of the ten games in my weekly ESPN college football pool. The site shows the pick, estimated win probability, team stats and the reasons behind it.

[View the picks](https://dstaggs-15.github.io/CFBPICKEM26/) · [Season record](https://dstaggs-15.github.io/CFBPICKEM26/record.html) · [Separate rankings](https://dstaggs-15.github.io/cfbranking/)

## How the model decides

The live model is `joint-stats-elo-v3`. **Current-season stats and Elo work together in one fitted model.** There isn't an Elo pick followed by a separate stats vote. Both enter the same calculation, and historical game outcomes determine their weights.

First, the model estimates every team's offense and defense from the current season's completed FBS games. It solves those ratings together, accounting for each team's opponents and home field. A good offensive game against a strong defense means more than the same performance against a weak defense. Ridge regression keeps small samples from producing extreme ratings.

For an upcoming game, it compares the home offense against the away defense and the away offense against the home defense. It makes these comparisons for PPA, success rate, explosiveness and scoring. These are matchup edges, not a team's raw national rank.

Next, regularized logistic regression combines those four edges with pregame Elo, venue, rest and postseason context. It learns from historical games whose inputs were built using information available before those games. The final scores are the training labels; they never enter their own pregame profiles.

The result is one probability. The model picks the home team at 50% or higher; otherwise it picks the away team. Your separate rankings can make the tiny adjustment described below.

## The inputs

| Input | What it tells the model |
| --- | --- |
| PPA matchup edge | Expected difference in play efficiency after accounting for both offenses, both defenses and venue |
| Success-rate matchup edge | Difference in how consistently the two offenses should succeed against these defenses |
| Explosiveness matchup edge | Difference in productive big-play performance against these opponents |
| Scoring matchup edge | Difference in scoring performance after accounting for opponents and venue |
| Elo | Strength estimated from earlier wins and losses, with rating gaps cut in half each offseason |
| Venue | Whether the game is at a home stadium or a neutral site |
| Rest | Difference in days since each team's previous scheduled game |
| Postseason | Whether this is a postseason game |

PPA estimates how much a play changes expected scoring value. The scoring edge is a statistical input, not a published point-spread forecast. Scoring includes all points in the final score; it isn't a play-level offense-only measure.

There is no Alabama bonus, team-name feature or betting-line input in the winner model. Team names connect games inside the current season's opponent-adjustment equations, but the model that learns historical winners doesn't receive those names.

## The weights

The model learns the weights together. It doesn't use a manually assigned recipe such as 60% Elo and 40% stats.

This is the September 30, 2026 fit. Inputs are scaled using the training data. A coefficient is the score change for a one-standard-deviation increase in an input. **It is not a percentage of the prediction or a percentage-point change in win probability.**

| Input | Fitted coefficient |
| --- | ---: |
| Elo home win estimate | +0.6499 |
| Success-rate matchup edge | +0.4520 |
| Scoring matchup edge | +0.3630 |
| Neutral-site indicator | +0.1049 |
| Explosiveness matchup edge | +0.1033 |
| PPA matchup edge | −0.0979 |
| Rest-day difference | −0.0278 |
| Postseason indicator | +0.0237 |

Elo has the largest individual coefficient. The statistical inputs have their own fitted contributions and can collectively outweigh it. The size of each contribution also depends on that game's input: a large coefficient does little when the matchup is close on that measure.

The inputs overlap. The negative PPA coefficient does **not** mean better efficiency is bad. It is the remaining association after the model also accounts for success, explosiveness, scoring and Elo. The weights describe a prediction formula, not cause and effect.

The model starts with an intercept of +0.4091, then adds its weighted inputs. Missing statistical inputs are filled using training-data averages and get explicit missingness flags. Each flag has a coefficient of about +0.0071 in this fit.

The total is a score `z`. The model converts it to a probability using `P(home win) = 1 / (1 + exp(-z))`. The card's explanations identify actual fitted terms supporting the pick and terms pulling toward the other team. The prediction JSON includes those terms, the intercept and the input values so the calculation can be checked.

## What carries over from last season?

**The four statistical profiles use the current season only.** They restart each season. No previous-season efficiency prior feeds v3.

Elo still carries earlier results, with its rating gaps halved each offseason. That makes it useful when the current season has little data, but it can also favor a team whose current stats are less convincing. Its contribution is visible in the explanation.

Statistical snapshots use completed FBS games before the upcoming game's week begins on Monday at 00:00 UTC. Both teams need at least two earlier FBS games. Before that, statistical edges are marked missing. Training learns how to handle missing profiles alongside Elo and context.

Future games aren't observations. Neither the current game's score nor another result later that week can change its stored pregame profile. Mature profiles must pass coverage checks so a broken stats feed can't quietly become an average team.

## Historical comparisons

Historical outcomes affect the prediction by training the combined model's weights across all eligible training games.

The 75 similar games on a card are examples of comparable pregame profiles. They now use the four joint statistical edges **and Elo**, with the same home/neutral venue type. Team names, betting lines and outcomes don't determine similarity. Outcomes are counted after the neighbors are chosen, and the three closest games are shown.

Their win rate is still descriptive. It isn't added to the prediction as another vote. The tested 75-neighbor classifier was less accurate than the joint model. A selected sample can disagree with the fitted forecast; the model learns from more than those 75 games.

## What testing showed

Each tested season's winner model was trained only on older seasons. On 6,800 FBS games from 2017–2025:

| Model | Correct winners | Brier error, lower is better |
| --- | ---: | ---: |
| Previous statistical model, v2 | 70.44% | 0.19370 |
| Current joint stats-and-Elo model | 70.31% | 0.19103 |
| Joint stats without Elo | 68.68% | 0.19899 |
| 75 similar joint profiles, without Elo | 68.00% | 0.20411 |

The joint model improved probability error but was nearly tied, slightly lower, in winner accuracy. A second comparison selected shrinkage using only earlier held-out seasons and showed the same tradeoff. We promoted the joint model to make the current-season matchup statistics and Elo work together as requested, supported by its better probability scores. **We are not claiming it improved the number of correct picks.**

The original pool record is 21–19 through four weeks. These all-FBS tests do not establish 73–75% accuracy on the ten-game ESPN pool. The research is retrospective and has been used during development; future frozen picks remain the prospective test.

[Full joint-model research and results](JOINT_PROFILE_RESEARCH.md) · [Earlier research](PROFILE_RESEARCH.md)

## Rankings, betting lines and news

The numbers beside teams come from your separate computer-ranking site, not an official poll. Its scores can adjust a prediction only when both teams appear in a valid, recent snapshot published before kickoff. Missing teams aren't assigned an invented rank.

The adjustment is `0.04 × (home score − away score)`, capped at `±0.0005` probability: **±0.05 percentage points**. A 60.00% estimate can move at most to 60.05% or 59.95%. This small adjustment hasn't been validated historically. The rankings website itself was not changed, and its statistical tables do not feed the model.

Betting lines are comparisons and inputs to the upset-watch display. They don't enter the winner model. News doesn't change the probability. Injuries and roster news aren't trained prediction inputs yet. The card's raw team-stat ranks are descriptive CFBD averages; the model uses adjusted pregame matchup edges.

## Updating a week

Upset Watch now tests each model-picked underdog against a historical research
screen. The 2026 conditions are at least 55% model probability, a spread of 3
points or less, and five prior FBS games per team. **This screen is not proven:**
choosing rules on older seasons and testing the next season returned 46.7% on
272 picks. Matching it gets a research label, not a strong recommendation.
The main pick remains unchanged. [Upset test details](UPSET_RESEARCH.md).

1. Edit `docs/input/games.txt` with the ten ESPN matchups, one per line: `Away @ Home` or `Team A vs Team B` for a neutral-site game.
2. In GitHub Actions, open **Run weekly pipeline** and select **Run workflow**. The normal season range is `2014-2026`.
3. The job tests, fetches CFBD data, grades completed archives, builds profiles, trains and backtests the joint model, then publishes picks and news.

The CFBD key stays in the repository's `CFBD_API_KEY` secret. Predictions are refused after kickoff. Rerunning before kickoff can refresh the board but can't replace a version's first archived picks.

Original archives remain in `historicals/predictions/`. Each model's archive lives under `historicals/model_versions/`, including `statistical-logistic-v2` and `joint-stats-elo-v3`. The record page grades them separately once every game on a slate has finished. Original results aren't rewritten by a model change. Weeks 1–2 have reported totals without individual archived picks.

Optional ESPN crowd shares go in `docs/input/crowd_picks.json`; stale snapshots are ignored. **Refresh news** can run separately.

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

Set `CFBD_API_KEY` before fetching. `joint_profile_audit.py` reproduces the research comparison. Older drive, boosted-tree and ensemble candidates remain available for audits; they don't feed the live v3 picks.

The weekly build writes `data/derived/joint_training.parquet`. Older research scripts use `training.parquet`, so rerunning an old audit doesn't replace the live model's input table.
