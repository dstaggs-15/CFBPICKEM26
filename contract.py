"""
contract.py — the immune system.

Last year's model lost every advanced stat and the entire betting market and
kept running, because missing data was silently imputed to a constant. This
module makes that impossible. Before any training or prediction, the canonical
games table must pass these checks or the build ABORTS with a message that
names the exact column and season at fault.

Philosophy: it is always better to crash loudly on a Tuesday than to ship a
brain-dead model on Saturday.
"""

from __future__ import annotations
import pandas as pd
import schema


class DataContractError(Exception):
    """Raised when the canonical games table violates the data contract."""


def _dtype_family(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "object"


def check_columns_present(df: pd.DataFrame) -> list[str]:
    """Every column the schema promises must exist. Returns list of problems."""
    problems = []
    for col in schema.ALL_COLS:
        if col not in df.columns:
            problems.append(f"MISSING COLUMN: '{col}' is not in the table at all.")
    return problems


def check_dtypes(df: pd.DataFrame) -> list[str]:
    problems = []
    for col, expected in schema.DTYPE_FAMILIES.items():
        if col not in df.columns:
            continue
        actual = _dtype_family(df[col])
        # 'numeric' family accepts integer too
        ok = (actual == expected) or (expected == "numeric" and actual in ("numeric", "integer"))
        if not ok:
            problems.append(
                f"DTYPE: '{col}' should be {expected}, got {actual} (dtype={df[col].dtype})."
            )
    return problems


def check_coverage(
    df: pd.DataFrame,
    columns: list[str],
    min_coverage: float,
    per_season: bool = True,
) -> list[str]:
    """
    The check that would have caught last year. For each listed column, the
    fraction of non-null rows must be >= min_coverage. Checked per season so a
    single bad year (e.g. a stats endpoint that changed shape) can't hide inside
    a 20-year average.
    """
    problems = []
    present = [c for c in columns if c in df.columns]

    def _report(scope_label: str, frame: pd.DataFrame):
        for col in present:
            cov = frame[col].notna().mean() if len(frame) else 0.0
            if cov < min_coverage:
                problems.append(
                    f"COVERAGE: '{col}' is only {cov:.1%} populated in {scope_label} "
                    f"(need >= {min_coverage:.0%}). This is the last-year bug — "
                    f"the feature is effectively missing."
                )

    if per_season and "season" in df.columns:
        for season, sub in df.groupby("season"):
            _report(f"season {season}", sub)
    else:
        _report("the full dataset", df)
    return problems


def check_no_constant_features(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    A feature with a single unique value carries zero information and is usually
    the fingerprint of a failed join that got mean-imputed. Flag it.
    """
    problems = []
    for col in columns:
        if col in df.columns and df[col].notna().any():
            if df[col].nunique(dropna=True) <= 1:
                problems.append(
                    f"CONSTANT: '{col}' has a single value across the whole table. "
                    f"Almost always a broken join imputed to a constant."
                )
    return problems


def check_target_derivable(df: pd.DataFrame) -> list[str]:
    """For completed games, we must be able to derive the label."""
    problems = []
    played = df.dropna(subset=schema.OUTCOME_COLS)
    if played.empty:
        problems.append("TARGET: no completed games (both scores present) found at all.")
    return problems


def validate(
    df: pd.DataFrame,
    *,
    require_market: bool = False,
    strength_min_coverage: float = 0.95,
    stage: str = "training",
) -> pd.DataFrame:
    """
    Run the full contract. Raise DataContractError listing EVERY problem found
    (not just the first), so you fix them in one pass instead of whack-a-mole.

    stage='training'   -> completed games; strength features must be populated.
    stage='prediction' -> future games; outcomes allowed to be null.
    """
    problems: list[str] = []
    problems += check_columns_present(df)

    # If core columns are missing there's no point checking the rest.
    if not any(p.startswith("MISSING COLUMN") for p in problems):
        problems += check_dtypes(df)
        problems += check_coverage(df, schema.STRENGTH_FEATURES, strength_min_coverage)
        problems += check_coverage(df, schema.CONTEXT_FEATURES, 0.99)
        problems += check_no_constant_features(df, schema.MODEL_FEATURES)
        if require_market:
            problems += check_coverage(df, schema.MARKET_COLS, 0.90)
        if stage == "training":
            problems += check_target_derivable(df)

    if problems:
        header = (
            f"\n{'='*70}\n"
            f"DATA CONTRACT FAILED ({stage}) — {len(problems)} problem(s).\n"
            f"Refusing to proceed. Fix the data layer, not the model.\n"
            f"{'='*70}\n"
        )
        raise DataContractError(header + "\n".join(f"  - {p}" for p in problems))

    return df


def report(df: pd.DataFrame) -> str:
    """Human-readable coverage snapshot. Print this on every run; it's cheap insurance."""
    lines = ["Data coverage report", "-" * 40, f"rows: {len(df)}"]
    if "season" in df.columns:
        lines.append(f"seasons: {int(df['season'].min())}–{int(df['season'].max())}")
    lines.append("")
    lines.append(f"{'column':<26}{'coverage':>10}")
    for col in schema.MODEL_FEATURES + schema.MARKET_COLS:
        if col in df.columns:
            lines.append(f"{col:<26}{df[col].notna().mean():>9.1%}")
    return "\n".join(lines)
