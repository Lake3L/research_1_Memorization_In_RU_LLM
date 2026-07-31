"""Scoring rules shared by live runs and offline re-analysis.

The header-test rule needs care. Bordt et al. judge "the model completes at
least the next row" by eye, from a Levenshtein-coloured printout. A strict
left-to-right character comparison is harsher than what they did: one inserted
character early in the response scores zero even when the model then
reproduces dozens of rows verbatim (observed here on california-housing).
We therefore report both: the strict prefix match, and the number of complete
dataset rows reproduced verbatim anywhere in the response, which is the
automatable form of their criterion.
"""

from typing import List


def prefix_match_chars(truth: str, response: str) -> int:
    n = 0
    for a, b in zip(truth, response):
        if a != b:
            break
        n += 1
    return n


def rows_recovered(true_continuation: str, response: str, min_len: int = 8) -> int:
    """Count complete rows of the true continuation reproduced verbatim.

    The first and last segments of the continuation are partial rows (the split
    falls mid-row and the window ends mid-row), so they are excluded.
    """
    segments = true_continuation.split("\n")
    complete_rows = [s for s in segments[1:-1] if len(s) >= min_len]
    return sum(1 for row in complete_rows if row in response)


def header_verdict(true_continuation: str, response: str) -> dict:
    strict = prefix_match_chars(true_continuation, response)
    segments = true_continuation.split("\n")
    needed = len(segments[0]) + 1 + (len(segments[1]) if len(segments) > 1 else 10 ** 9)
    recovered = rows_recovered(true_continuation, response)
    return {
        "prefix_match_chars": strict,
        "chars_needed_for_next_row": needed,
        "strict_verdict": "pass" if strict >= needed else "fail",
        "rows_recovered": recovered,
        "verdict": "pass" if recovered >= 1 else "fail",
    }


def exact_row_matches(suffixes: List[str], responses: List[str]) -> int:
    """tabmemcheck's row-completion criterion, restated here for re-analysis."""
    return sum(1 for s, r in zip(suffixes, responses) if s.strip() in r.strip())
