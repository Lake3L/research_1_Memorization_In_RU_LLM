"""Print one run's result file as a table: what was loaded, what was measured.

Every column that AMENDMENT_5 §1 requires beside a count travels with it here —
the baseline the rate is tested against, the exact p-value, the smallest rate the
cell could have excluded, and the digits-per-row covariate — so that a zero is
never read without the two numbers that say what it means.

Usage:
  python src/report_run.py results/gateA_*.json
  python src/report_run.py results/gateA_*.json --brief
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt(value, spec, dash="-"):
    if value is None:
        return dash
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def report(path, brief=False):
    data = json.load(open(path, encoding="utf-8"))
    load = data.get("load", {}) or {}
    gate = data.get("gate", {})

    print("=" * 104)
    print(f"{data['model']}   prompting={data.get('prompting', 'chat')}  "
          f"protocol={data.get('protocol', 'legacy')}  plan={data.get('plan')}")
    print(f"  revision   : {data.get('revision_loaded')}")
    print(f"  loaded     : quantized={load.get('quantized')} "
          f"{load.get('memory_footprint_gb')} GB  attn={load.get('attn_implementation')}  "
          f"devices={load.get('devices')}")
    instrument = data.get("instrument_check", {})
    print(f"  instrument : perfect "
          f"{instrument.get('perfect_memorizer', {}).get('matches')}/"
          f"{instrument.get('perfect_memorizer', {}).get('n')}, echo "
          f"{instrument.get('format_echo', {}).get('matches')}/"
          f"{instrument.get('format_echo', {}).get('n')} -> "
          f"{'OK' if instrument.get('instrument_ok') else 'BROKEN'}")
    print(f"  verdict    : {gate.get('verdict')}  complete={gate.get('complete')} "
          f"({gate.get('cells_run')}/{gate.get('cells_planned')} cells)")
    if brief:
        return data

    head = (f"  {'test':11s} {'dataset':22s} {'result':>9s} {'rate':>6s} {'base':>7s} "
            f"{'p':>9s} {'min det':>8s} {'dig/row':>8s} {'wf':>5s} {'near':>5s} "
            f"{'lev':>5s} {'sec':>6s}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in data["results"]:
        if "error" in r:
            print(f"  {r.get('test', '?'):11s} {r.get('dataset_key', '?'):22s} "
                  f"{'ERROR':>9s}  {str(r['error'])[:56]}")
            continue
        if r.get("test") == "header":
            result = f"{r.get('verdict')}({r.get('rows_recovered', 0)}r)"
        else:
            result = f"{r.get('matches')}/{r.get('n')}"
        print(f"  {r['test']:11s} {r['dataset_key']:22s} {result:>9s} "
              f"{fmt(r.get('rate'), '6.3f')} {fmt(r.get('baseline_rate'), '7.4f')} "
              f"{fmt(r.get('p_value'), '9.2e')} "
              f"{fmt(r.get('minimum_detectable_rate'), '8.3f')} "
              f"{fmt(r.get('digits_per_row'), '8.1f')} "
              f"{fmt(r.get('well_formed_rate'), '5.0%')} "
              f"{fmt(r.get('near_match_rate'), '5.0%')} "
              f"{fmt(r.get('mean_normalized_levenshtein'), '5.2f')} "
              f"{fmt(r.get('seconds'), '6.0f')}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--brief", action="store_true")
    args = ap.parse_args()

    paths = []
    for pattern in args.paths:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    for path in paths:
        report(path, brief=args.brief)
    return 0


if __name__ == "__main__":
    sys.exit(main())
