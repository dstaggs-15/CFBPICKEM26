"""
demo.py — prove the whole Layer 1 works, with no API key.

Run from repo root:  python demo.py
"""
import contract, backtest
from baselines import EloModel, MarketModel
from model import V1Model
from make_synthetic import make


def main():
    print("\n### 1. Generate structurally-honest synthetic data")
    df = make()
    print(f"    {len(df)} games, seasons {df['season'].min()}–{df['season'].max()}")

    print("\n### 2. Run the data contract (the immune system)")
    contract.validate(df, require_market=True, stage="training")
    print("    PASSED. Coverage snapshot:\n")
    print(contract.report(df))

    print("\n### 3. Walk-forward backtest: model vs the baselines it must beat")
    factories = {
        "model": lambda: V1Model(),
        "elo": lambda: EloModel(),
        "market": lambda: MarketModel(),
    }
    per_fold, summary, disagree = backtest.walk_forward(df, factories)
    print(backtest.format_report(summary, disagree))

    print("\n\n### 4. Now prove the contract catches LAST YEAR'S EXACT BUG")
    print("    (a feature join silently returns all-null)\n")
    broken = make(break_column="off_ppa_adj_diff")
    try:
        contract.validate(broken, require_market=True, stage="training")
        print("    !!! contract failed to catch it — that would be bad")
    except contract.DataContractError as e:
        print(str(e))


if __name__ == "__main__":
    main()
