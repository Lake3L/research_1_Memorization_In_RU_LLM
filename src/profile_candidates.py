"""Profile candidate files before any of them is frozen or shown to a model.

Every number a dataset amendment cites for a candidate comes from here, so
that the choice among candidates can be audited: nothing is measured on a
model, and nothing depends on a model's answer. Per file:

  shape        encoding, separator, physical lines vs parsed records, columns,
               Cyrillic in header / in values, a second header line (the
               data.mos.ru export carries an English and a Russian header);
  windows      header + first row against the 500-character header window;
               characters per row, digits per row (the entropy covariate);
  R2 null      duplicate-row rate, the prefix-predictor rate over every
               window, the rule-of-three floor, the resulting null and the
               minimum detectable rate at 250 queries (AMENDMENT_7 R2);
  R1 share     near-duplicate windows at tau 0.05 / 0.10 / 0.20 / 0.30, on a
               seeded sample of windows (AMENDMENT_7 R1);
  order        share of consecutive rows whose first field is non-decreasing
               (a sorted file violates the row-order assumption of the test);
  columns      cardinality ratio per column (feature-completion candidates).

Usage:
  python src/profile_candidates.py FILE [FILE ...] --out profile.json [--max-lines 200000] [--sample 3000]
"""

import argparse
import csv
import io
import json
import os
import random
import re
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import infer_separator  # noqa: E402
from prefix_baseline import norm_lev, null_rate, predictor_rate, TAUS  # noqa: E402

PREFIX_ROWS = 8
HEADER_WINDOW = 500
QUERIES = 250
CYR = re.compile(r"[Ѐ-ӿ]")


def decode(raw: bytes):
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc), enc, raw.startswith(b"\xef\xbb\xbf")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8?", False


def records_of(text: str, sep: str):
    try:
        return list(csv.reader(io.StringIO(text), delimiter=sep))
    except csv.Error:
        return [line.split(sep) for line in text.splitlines()]


def min_detectable(p0: float, n: int = QUERIES) -> float:
    for k in range(n + 1):
        if stats.binomtest(k, n, p0, alternative="greater").pvalue < 0.05:
            return k / n
    return 1.0


def near_share_sampled(rows, sample, seed=0):
    """Minimum normalised distance from the ninth row to the eight before it,
    on `sample` windows drawn without replacement; the share at each tau."""
    windows = list(range(1, len(rows) - PREFIX_ROWS))
    if not windows:
        return {"windows_sampled": 0, "shares": {str(t): None for t in TAUS}}
    rng = random.Random(seed)
    picked = windows if len(windows) <= sample else rng.sample(windows, sample)
    mins = []
    for i in picked:
        truth = rows[i + PREFIX_ROWS]
        best = 1.0
        for row in rows[i:i + PREFIX_ROWS]:
            longest = max(len(truth), len(row), 1)
            if abs(len(truth) - len(row)) / longest > max(TAUS):
                continue
            best = min(best, norm_lev(truth, row))
            if best == 0.0:
                break
        mins.append(best)
    return {"windows_sampled": len(picked),
            "shares": {str(t): round(sum(m <= t for m in mins) / len(mins), 4) for t in TAUS}}


def profile(path: str, max_lines: int, sample: int) -> dict:
    raw = open(path, "rb").read()
    text, enc, bom = decode(raw)
    lines = text.splitlines()
    truncated = len(lines) > max_lines
    lines = lines[:max_lines]
    header = lines[0] if lines else ""
    sep = infer_separator(header) if header else ","
    recs = records_of("\n".join(lines), sep)
    n_cols = len(recs[0]) if recs else 0
    body_lines = [l for l in lines[1:] if l.strip()]
    body_recs = [r for r in recs[1:] if any(x.strip() for x in r)]
    second_header = (len(recs) > 1 and len(recs[1]) == n_cols
                     and not CYR.search(header) and bool(CYR.search(lines[1]))
                     and not re.search(r"\d", lines[1].replace('"', "")[:40]))

    dup = 1 - len(set(body_lines)) / len(body_lines) if body_lines else 0.0
    pred = predictor_rate(body_lines, PREFIX_ROWS)
    p0, floor = null_rate(dup, pred["rate"], pred["windows"])
    near = near_share_sampled(body_lines, sample)

    first_col = [r[0] for r in body_recs if r]
    def key(v):
        try:
            return (0, float(v.replace(",", ".")))
        except ValueError:
            return (1, v)
    pairs = list(zip(first_col, first_col[1:]))
    nondecreasing = (sum(key(a) <= key(b) for a, b in pairs) / len(pairs)) if pairs else None

    card = {}
    if body_recs and n_cols:
        for j, name in enumerate(recs[0]):
            col = [r[j] for r in body_recs if len(r) > j]
            card[name or f"col{j}"] = round(len(set(col)) / len(col), 4) if col else None

    row_len = [len(l) for l in body_lines]
    digits = [sum(ch.isdigit() for ch in l) for l in body_lines]
    return {
        "file": path,
        "bytes": len(raw), "encoding": enc, "bom": bom, "separator": sep,
        "physical_lines": len(text.splitlines()), "profiled_lines": len(lines),
        "truncated_to_max_lines": truncated,
        "records": len(body_recs), "body_lines": len(body_lines), "columns": n_cols,
        "malformed_records": sum(len(r) != n_cols for r in body_recs),
        "cyrillic_in_header": bool(CYR.search(header)),
        "cyrillic_in_values": any(CYR.search(l) for l in body_lines[:2000]),
        "second_header_line": second_header,
        "header_plus_first_row_chars": len(header) + 1 + (len(body_lines[0]) if body_lines else 0),
        "header_test_applicable": len(header) + 1 + (len(body_lines[0]) if body_lines else 0) <= HEADER_WINDOW,
        "chars_per_row_mean": round(sum(row_len) / len(row_len), 1) if row_len else None,
        "chars_per_row_max": max(row_len) if row_len else None,
        "digits_per_row_mean": round(sum(digits) / len(digits), 1) if digits else None,
        "prompt_chars_8_rows_mean": round(8 * sum(row_len) / len(row_len)) if row_len else None,
        "duplicate_rate": round(dup, 5),
        "predictor": {"rate": round(pred["rate"], 5), "windows": pred["windows"], "by_kind": pred["by_kind"]},
        "rule_of_three": round(floor, 5),
        "null_r2": round(p0, 5),
        "min_detectable_rate_250": round(min_detectable(p0), 4),
        "near_duplicate": near,
        "first_field_nondecreasing_share": round(nondecreasing, 4) if nondecreasing is not None else None,
        "cardinality_ratio": card,
    }


def table(profiles):
    head = ("| file | enc | sep | lines | records | cols | cyr hdr/val | hdr+row | chars/row | digits/row | dup | pred | null | MDR | near 0.10 | sorted | max card |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    out = [head]
    for p in profiles:
        cr = p["cardinality_ratio"]
        top = max(cr.items(), key=lambda kv: kv[1] or 0) if cr else ("", None)
        out.append("| {f} | {enc} | `{sep}` | {pl} | {rec} | {cols} | {ch}/{cv} | {hw} | {cpr} | {dpr} | {dup} | {pred} | {null} | {mdr} | {near} | {sort} | {top} |".format(
            f=os.path.basename(p["file"]), enc=p["encoding"], sep=p["separator"], pl=p["physical_lines"],
            rec=p["records"], cols=p["columns"], ch=int(p["cyrillic_in_header"]), cv=int(p["cyrillic_in_values"]),
            hw=p["header_plus_first_row_chars"], cpr=p["chars_per_row_mean"], dpr=p["digits_per_row_mean"],
            dup=p["duplicate_rate"], pred=p["predictor"]["rate"], null=p["null_r2"], mdr=p["min_detectable_rate_250"],
            near=p["near_duplicate"]["shares"].get("0.1"), sort=p["first_field_nondecreasing_share"],
            top=f"{top[0][:24]}={top[1]}"))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lines", type=int, default=200_000)
    ap.add_argument("--sample", type=int, default=3000)
    args = ap.parse_args()
    profiles = []
    for f in args.files:
        print(f"profiling {f}", file=sys.stderr, flush=True)
        profiles.append(profile(f, args.max_lines, args.sample))
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(profiles, fh, ensure_ascii=False, indent=2)
    print(table(profiles))


if __name__ == "__main__":
    main()
