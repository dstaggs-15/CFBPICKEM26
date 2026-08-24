# CFB Pick'em Model

A college-football prediction system that picks winners for a weekly ESPN pick'em pool — and, more importantly, **tells you how confident it is and why**, in plain English. It runs entirely on GitHub (no server, no cost) and publishes to a live website.

This is a ground-up rebuild of last year's version. The rest of this README explains what it is, how each part works, and the one question everyone asks: *how can it possibly know who'll win before a single game is played?*

---

## Why it was rebuilt (the short, honest version)

Last year's model looked finished — it had a website, confidence meters, the works — but under the hood it had quietly lost almost all of its data. The advanced team stats never loaded. The betting lines never loaded. Nothing warned anyone, because missing data was silently filled in with averages, so the model just ran on almost nothing and nobody could tell. On a fair test it was actually a hair *worse* than a one-line power rating.

The rebuild is designed around a single idea: **make that kind of silent failure impossible.** If a piece of data is missing, the whole thing stops and says so, by name, before it ever makes a pick. Better to break loudly on a Tuesday than to lie quietly on a Saturday.

---

## How it works, station by station

Think of it as an assembly line. Raw football data comes in one end; a pick with an explanation comes out the other. Each file is one station.

### 1. The blueprint — `schema.py`
A single list that defines exactly what one game's worth of data looks like: the teams, the score, the site, the stats, the betting line. Everything else in the project refers back to this one list, so there's never a disagreement about what the data should contain.

### 2. The inspector — `contract.py`
The most important file, and the thing last year didn't have. Before any data is allowed through, this checks it: are the stats actually here? Is any column suspiciously empty for a given season? Is anything a single frozen value (the fingerprint of a broken connection)? If something's wrong, it **stops the line and names the exact column and year.** This is the immune system.

### 3. The data puller — `fetch_cfbd.py`
The only file that reaches out to the internet. It pulls three things from the College Football Data API: game results, betting lines, and advanced team stats (efficiency numbers like PPA — points-per-play — and success rate). It saves them and prints a coverage report so you can *see* how complete the data is. This is where your `CFBD_API_KEY` is used.

### 4. The measuring sticks — `baselines.py`
Two dead-simple predictors: one based on **Elo** (a power rating that rises when you win and falls when you lose) and one based on the **betting line**. These aren't the model — they're the bar the model has to clear. If the fancy model can't beat "just trust the betting line," it isn't worth running. Last year had no such bar, which is why nobody noticed the model was underperforming.

### 5. The feature builder — *(next to be built)*
Turns raw stats into things the model can actually learn from. The headline one is **opponent-adjusted strength**: "Team X averaged 0.6 points per play" is meaningless until you adjust for *who they played*. Beating a great defense at 0.6 is elite; doing it against a cupcake is not. This adjustment is the single feature that separates two evenly-matched teams — and it's what vanished entirely last year.

### 6. The brain — `model.py`
A gradient-boosting model that learns from the football features only. Two rules baked in: it is **not** allowed to peek at the betting line (the line is the benchmark, not an input — otherwise it's just parroting Vegas), and it checks its own confidence honestly on data it hasn't seen, instead of grading its own homework the way last year's did.

### 7. The report card — `backtest.py`
Answers "how good is this, really?" honestly: it trains on old seasons and tests on a season it has never seen, exactly how it'll face the future. It prints your model beside the two baselines, and — the part that matters most — it splits games into where the model **agrees** with the betting line and where it **disagrees**. Agreeing is free; anyone can pick the favorite. The only place the model can earn its keep is winning the games where it *disagrees*. That row is the whole ballgame.

### 8. The weekly slate — `weekly_input.py` + `docs/input/games.txt`
Each week you edit `games.txt` with ESPN's 10 games, one per line (`Away @ Home`, or `Away vs Home` for a neutral site). Commit it, and the pipeline predicts exactly those games.

### 9. The website — `docs/`
A static site (plain HTML/CSS/JS) served free by GitHub Pages. Each game is a card with a **tug-of-war meter** built from the two teams' real colors — the more confident the pick, the more the bar is filled by that team's color. Each card shows the pick, how it compares to the betting line (an AGREE/DISAGREE tag), and a slot for the plain-English explanation and (later) an AI second opinion.

---

## The question everyone asks: how does it know before any games are played?

Say it's Week 1 and the model gives Ole Miss a 68% chance to win. No games have happened yet — so what on earth is that number based on? Fair question. Here's the honest answer.

**Before the season starts, the model leans on what carries over from last year:**
- **Where each team ended last season** (their final power rating). A team that finished strong starts strong; this "memory" fades a bit over the off-season but doesn't reset to zero.
- **How good each program usually is.** Blue-bloods and perennial contenders get the benefit of the doubt over programs that are usually rebuilding.
- **Home field and travel.** Playing at home is worth real points; a long road trip costs something. A neutral-site game (the `vs` games) removes that edge.

So a Week 1 "68%" is really the model saying: *based on how these two teams finished last year, who's generally stronger, and who's at home, Ole Miss should win about two times out of three.* It is a starting estimate from prior evidence, not a guess pulled from nowhere — the same way you'd expect a returning playoff team to beat a rebuilding one even though you haven't watched them yet.

**As the season plays, the memory gets replaced by this year's reality.** After a few weeks the model shifts its weight onto how each team is *actually* performing right now — their recent efficiency, adjusted for the quality of who they've played. By midseason, last year barely matters; the number is grounded in games you've watched. That's also when the model is at its sharpest.

**And this is exactly why the "why" panel matters.** When it's built, clicking a game will show the handful of things pushing the pick — "Ole Miss rated higher entering the year," "playing at a neutral site (no home edge for either)," "stronger recent efficiency" — so you're never asked to just trust a naked percentage. If the model can't explain itself in plain terms, you shouldn't believe it, and neither should we.

---

## The weekly routine (once it's all wired up)

1. Edit `docs/input/games.txt` with this week's 10 ESPN games. Commit.
2. The GitHub Action runs automatically: pulls fresh data, rebuilds features, predicts your 10 games, writes `docs/predictions.json`.
3. Your website updates itself with the new picks, meters, and explanations.

No computer setup required — it all happens in your browser and in GitHub's cloud.

---

## Running it in the browser (no local install)

Everything runs through GitHub Actions, so you never install Python on your own machine.

- **Your API key** lives as a repository secret: **Settings -> Secrets and variables -> Actions -> New repository secret**, named `CFBD_API_KEY`.
- **To fetch data**, go to the **Actions** tab -> **Fetch CFBD data** -> **Run workflow**.
- **The website** is turned on at **Settings -> Pages -> Deploy from a branch -> `main` / `docs`**.

---

## What's built vs. what's coming

**Built and working:**
- The data contract (the inspector)
- Real data fetching from CFBD, with coverage reporting
- Elo and betting-line baselines
- The walk-forward backtest with the agree/disagree analysis
- The weekly `games.txt` slate reader
- The live website with team-colored confidence meters

**Coming next, in order:**
1. **The feature builder + trained model** — turns the raw data into real, opponent-adjusted predictions. Unlocks everything below.
2. **The "why" panel** — click a game to see, in plain English, the real factors behind the pick (only possible once the model exists, so it shows genuine reasoning, not filler).
3. **Team stats in the card** — recent efficiency, form, and the betting line, so you can sanity-check the pick yourself.
4. **News per game** — headlines and injury notes pulled in weekly and attached to each matchup.
5. **The AI second opinion** — an independent, plain-English take that gets scored on the same backtest as the model, so it has to earn its credibility too.

---

## A note on honesty

The guiding rule of this project: **never show a number or a pick without a real basis behind it.** Placeholder data is always labeled as placeholder. The model doesn't ship unless it beats the baselines on a fair test. The AI opinion, when it arrives, keeps a public win-loss record. If something can't be explained or measured, it doesn't go on the site. That discipline is the entire difference between this and last year.
