"""
weekly_input.py — your ESPN pick'em slate, one file a week.

Each week ESPN picks ~10 games. You list them in docs/input/games.txt (one per
line), commit, and the weekly workflow predicts exactly those games and updates
the site. This file just reads and sanity-checks that list.

games.txt format (forgiving on purpose):
    # lines starting with a hash are comments, ignored
    # blank lines ignored
    LSU @ Vanderbilt          <- away team @ home team
    Georgia Tech @ Duke
    Louisville vs Ole Miss    <- use 'vs' for a neutral-site game (no home edge)

Team names should match how CFBD spells them (e.g. "Ole Miss", "Florida State").
The predict step will alias-correct common variants, but closer is better.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

GAMES_TXT = "docs/input/games.txt"


@dataclass
class Matchup:
    away_team: str
    home_team: str
    neutral: bool

    def __str__(self):
        joiner = "vs" if self.neutral else "@"
        return f"{self.away_team} {joiner} {self.home_team}"


class GamesFileError(Exception):
    pass


def parse_line(line: str) -> Matchup | None:
    """Return a Matchup, or None for comment/blank lines. Raise on malformed."""
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None

    # Neutral site if joined by ' vs ', home game if joined by ' @ '.
    neutral = False
    if " vs " in f" {raw} ".replace("VS", "vs"):
        parts = _split_on(raw, "vs")
        neutral = True
    elif "@" in raw:
        parts = _split_on(raw, "@")
    else:
        raise GamesFileError(
            f"Can't read this line (need '@' or 'vs' between teams): {raw!r}"
        )

    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise GamesFileError(f"Need exactly two teams on this line: {raw!r}")

    return Matchup(away_team=parts[0], home_team=parts[1], neutral=neutral)


def _split_on(raw: str, token: str) -> list[str]:
    # Split on the token as a whole word-ish separator, tolerate extra spaces.
    lowered = raw.lower()
    idx = lowered.find(f" {token} ")
    if idx == -1:
        # also allow no-space forms like "A@B"
        idx = lowered.find(token)
        return [raw[:idx].strip(), raw[idx + len(token):].strip()]
    return [raw[:idx].strip(), raw[idx + len(f" {token} "):].strip()]


def load_slate(path: str = GAMES_TXT) -> list[Matchup]:
    p = Path(path)
    if not p.exists():
        raise GamesFileError(
            f"{path} not found. Create it and list this week's games, one per line."
        )
    games, problems = [], []
    for n, line in enumerate(p.read_text().splitlines(), start=1):
        try:
            m = parse_line(line)
            if m:
                games.append(m)
        except GamesFileError as e:
            problems.append(f"  line {n}: {e}")

    if problems:
        raise GamesFileError("Problems in games.txt:\n" + "\n".join(problems))
    if not games:
        raise GamesFileError(f"{path} has no games in it yet.")

    # Gentle nudge, not an error — ESPN slates are usually 10.
    if len(games) != 10:
        print(f"  note: {len(games)} games found (ESPN pools are usually 10).")
    return games


if __name__ == "__main__":
    for m in load_slate():
        print(m)
