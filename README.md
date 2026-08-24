# CFB Pick'em Model — 2026 rebuild (Layer 1)

A from-scratch rebuild of the college-football pick'em model. This is **Layer 1: the core** — the data-quality contract, the baselines, and the walk-forward backtest harness. The fancy stuff (rich bubbles, LLM second opinion) comes later and plugs into this.

## Why a rebuild

Last year's model silently lost every advanced stat and the entire betting market to broken joins, then mean-imputed the holes to constants and kept running. On a proper walk-forward backtest it scored *below* a one-line Elo formula. Nothing in the code measured that, so it stayed invisible all season.

This rebuild is organized around one idea: **make that failure impossible to hide.**

## The three disciplines

1. **Fail loud, never impute silently.** `src/contract.py` validates the canonical games table before any training. If a feature is under-covered, constant, or missing — for any single season — the build aborts and names the column. Fix the data, not the model.

2. **Backtest-first.** `src/backtest.py` trains on past seasons, predicts the next, and always reports your model beside Elo and the market on the same games. The model doesn't ship unless it beats both — and the report ends with the only number that matters for a pick'em pool: **when the model disagrees with the market, does it win those games?**

3. **Market is the benchmark, not a feature.** The v1 model (`src/model.py`) trains only on football features and gets *scored against* the market. No circular explanations. Calibration is fit on a held-out season, not the training rows (last year's leak).

## Layout

```
src/schema.py      canonical column definitions — single source of truth
src/contract.py    the immune system (coverage / dtype / constant checks)
src/baselines.py   EloModel, MarketModel (the bar to clear)
src/model.py       V1Model — gradient boosting, leak-free calibration
src/backtest.py    walk-forward harness + disagreement analysis
scripts/fetch_cfbd.py  <- your CFBD_API_KEY plugs in here (only outside-world file)
scripts/make_synthetic.py  fake-but-honest data to test with no key
demo.py    runs the whole thing end to end
```

