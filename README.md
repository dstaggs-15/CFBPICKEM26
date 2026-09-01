# CFB Pick'em Model

A tool that predicts college football games for a weekly ESPN pick'em pool, shows a live website with the picks, and explains *why* it made each one in plain English. It runs entirely for free on GitHub — no server, no monthly cost.

This README explains what it does and how, in plain language, from the ground up.

---

## The one-sentence version

Every week you type in your 10 ESPN games, click one button, and a few minutes later your website shows real predictions — with a confidence percentage, a comparison to the Vegas betting line, each team's stats, and news headlines — for all 10 games.

---

## Why this exists (the honest backstory)

An earlier version of this looked done — website, percentages, the works — but it turned out to be quietly broken. It was supposed to use detailed team stats and betting lines, but those never actually loaded, and nothing told anyone. So it was making picks almost blind. When actually measured, it did *worse* than just picking whichever team was favored.

This version is built around one rule: **if data is missing, stop and say so loudly, instead of quietly guessing.** Every piece described below follows that rule.

---

## The pieces, in plain English

Think of this as a factory line. Each station does one job and hands off to the next.

**1. Get the data.** A script reaches out to a college football data service and downloads years of game results, betting lines, and detailed team stats (how efficient each team's offense and defense are, not just win/loss).

**2. Check the data.** Before anything is trusted, an "inspector" checks it — is anything suspiciously missing? Is any column just one repeated value (a sign something broke)? If so, it stops and names the exact problem instead of quietly continuing. This is the safeguard the old version never had.

**3. Turn stats into "who's actually better."** Raw stats aren't fair on their own — a team that scores a lot against weak opponents isn't the same as a team that scores a lot against strong ones. This step adjusts every team's numbers for *who they actually played*, so the comparisons are fair.

**4. Set the bar to beat.** Two simple, honest predictors are built first: one based on a power-rating system (Elo, like chess rankings), and one based on the Vegas betting line. These aren't fancy, but they're a fair baseline. The real model isn't allowed to call itself "good" unless it beats both of these.

**5. Train the actual model.** A machine-learning model studies years of past games — using the fair, adjusted stats — and learns the patterns that lead to wins.

**6. Grade it honestly.** The model is tested the fair way: it only ever predicts games it hasn't seen the result of yet (like predicting next season using only past seasons). It's scored against the two baselines from step 4. Most importantly, it's split into two groups: games where it *agrees* with Vegas, and games where it *disagrees*. Agreeing with Vegas is easy — anyone can do that. The only place this model can prove it's actually smart is in the disagreements. That number is the real report card.

**7. You type in this week's games.** Each week you edit one small text file with the 10 games from your ESPN pool (just team names, like `LSU @ Vanderbilt`).

**8. It makes the picks.** The trained model looks at exactly those 10 games and predicts a winner and a confidence percentage for each.

**9. It explains itself.** For every pick, it writes a short, plain-English reason — like "Tulane has been more efficient on offense against similar defenses" or "Vanderbilt is at home, which is worth a few points." It's built from the model's real inputs, not made up after the fact.

**10. It gathers news.** For each of the 20 teams playing this week, it searches for a few recent headlines (injuries, roster news, storylines) and attaches them to that game.

**11. It builds the website.** All of this — picks, percentages, stats, explanations, news — gets written into your live website automatically.

---

## What the website actually shows you

Each game is a card. Tap it and it expands to show:

- **The pick and the percentage** — a bar that fills with each team's real school colors, showing how confident the model is.
- **Whether it agrees with Vegas** — an AGREE or DISAGREE tag next to the betting line.
- **Why it picked that team** — a few plain-English bullet points.
- **Each team's season stats** — offense, defense, and how they rank nationally (e.g. "7th out of 134").
- **Recent news** — a few headlines about each team.
- An **AI take** slot, currently empty — reserved for a second, independent AI opinion (not built yet, see below).

If a game can't be matched to the schedule (usually because of a small spelling difference), the card says "not found" honestly instead of pretending to have an answer.

---

## How it knows anything before the season starts

Fair question — if no games have been played yet, what is a "68% chance" even based on? Three things:

1. **How each team finished last season** — a power rating that carries over and fades slowly, not a full reset.
2. **How good each program usually is** — a blue-blood program gets more benefit of the doubt than a team that's usually rebuilding.
3. **Home field** — playing at home is worth real points; a neutral-site game removes that edge.

As real games get played, the model leans more and more on *this season's* actual performance and less on last year's memory. By midseason, it's mostly grounded in games you've actually watched happen.

---

## Where your data comes from and where it goes

- **Game results, stats, and betting lines**: pulled from a service called CollegeFootballData, using your personal API key (kept private as a GitHub "secret," never visible in the code).
- **News headlines**: pulled from Google News' public feed for each team.
- **Everything runs on GitHub's computers**, not your own — you just click "Run workflow" in the Actions tab and it does the work in the cloud.
- **Your website**: a set of plain files (`docs/` folder) that GitHub hosts for free and updates every time the pipeline runs.

---

## The weekly routine

1. Edit the games file with this week's 10 ESPN matchups.
2. Go to the Actions tab, click **Run weekly pipeline**, click Run.
3. Wait a few minutes.
4. Your website updates itself with real picks, stats, and news for all 10 games.

Optionally, run **Refresh news** on its own (like Friday night) to get updated headlines without redoing the whole model.

---

## Honest status — what's real right now vs. what's still coming

**Actually working today:**
- Real data pulled from 2021-2025+ (tens of thousands of real games)
- The safety-check system that stops on bad data
- Fair, opponent-adjusted team stats
- A trained model, graded against Elo and the betting line
- The weekly picks pipeline, fully automated
- The live website with real team colors, expandable explanations, stats, and news

**Known rough edges:**
- A few teams occasionally show "not found" if their name doesn't exactly match the schedule's spelling
- Not every team has its exact brand color dialed in perfectly yet (falls back to a red/gray theme when unknown, so nothing looks broken)
- The full "does it actually beat Vegas" grade has only been checked on a shorter data window so far — a full-history check is the next big milestone, and the honest result so far shows the model doing well overall but not yet consistently beating Vegas when it disagrees with the line

**Not built yet:**
- Turnover stats (interceptions/fumbles) in the dropdown — currently only shows offense/defense efficiency
- The second, independent AI opinion next to the model's pick
- The Kalshi trading-focused version of this tool (a separate, future project)

---

## The one rule this whole project follows

**Never show a number without a real basis behind it.** If something can't be explained, measured, or verified, it doesn't go on the site. That discipline — not fancier math — is the actual difference between this version and the one that came before it.
