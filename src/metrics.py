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

import re
from typing import List

# A field that is a number and nothing else. Anything that does not match is left
# exactly as it is — an identifier, a name, a category — because "looks numeric"
# is not the same as "is a quantity".
_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)$")


def infer_separator(row: str) -> str:
    return max(",;\t", key=row.count)


def normalise_numbers(row: str, sep: str = None) -> str:
    """Canonicalise numeric fields so that formatting is not scored as content.

    Ward et al.'s caveat: `17.5` and `17.500` are the same quantity written two
    ways, and a string metric that cannot tell them apart will report a format
    difference as a memorization difference. That confound is fatal for H4, where
    the two arms differ in a factor (prompt language) that carries number formats
    with it, and it is present already in the canon — the published `uci-wine`
    ships `.28` where a pandas round-trip writes `0.28`.

    Applied only to the *approximate* metrics. The verbatim tests stay byte-exact,
    because that is the thing they are for: PREREGISTRATION.md §5 defines them on
    the published bytes and §7 forbids softening them.

    Fields are normalised individually after splitting on the row's separator, so
    that a decimal comma in a semicolon-separated file is handled without the two
    roles of the comma being confused.
    """
    sep = sep or infer_separator(row)
    out = []
    for field in row.split(sep):
        stripped = field.strip()
        candidate = stripped.replace(",", ".") if sep != "," else stripped
        if not _NUMBER.match(candidate):
            out.append(stripped)
            continue
        sign = "-" if candidate.startswith("-") else ""
        digits = candidate.lstrip("+-")
        if "." in digits:
            whole, frac = digits.split(".", 1)
            frac = frac.rstrip("0")
            whole = whole or "0"          # ".28" and "0.28" are one number
            digits = f"{whole}.{frac}" if frac else whole   # "22.0" and "22" too
        out.append(sign + digits)
    return sep.join(out)


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
