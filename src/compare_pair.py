"""Put a base model and its Russian adaptation side by side, cell by cell.

This is the shape of the H1b analysis (PREREGISTRATION.md §6, AMENDMENT_3 §2):
paired per dataset, on the verbatim counts as the primary outcome and on the mean
normalized Levenshtein distance as the secondary one. It is a reporting tool, not
a decision procedure — the Wilcoxon over datasets that H1b calls for needs the
full four-variant, three-seed design, and a pilot on one variant and one seed is
not that.

What it is for now is making the comparison visible without transcribing numbers
by hand, and making the cells that did not run visible as absences rather than as
blanks that read like zeros.

Usage:
  python src/compare_pair.py results/gateA_base.json results/gateA_adapted.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cells(path):
    data = json.load(open(path, encoding="utf-8"))
    by = {}
    for r in data["results"]:
        by[(r.get("test"), r.get("dataset_key"))] = r
    return data, by


def summarise(r):
    """One cell as a short string, with absence distinguished from zero."""
    if r is None:
        return "not run", None
    if "error" in r:
        kind = "OOM" if "OutOfMemory" in str(r["error"]) else "error"
        return kind, None
    if r["test"] == "header":
        return (f"{r.get('verdict')} ({r.get('rows_recovered', 0)}r)",
                r.get("rows_recovered"))
    return f"{r.get('matches')}/{r.get('n')}", r.get("rate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("adapted")
    ap.add_argument("--rescored-base", default=None)
    ap.add_argument("--rescored-adapted", default=None)
    args = ap.parse_args()

    base_data, base = cells(args.base)
    adapted_data, adapted = cells(args.adapted)

    def levs(path):
        if not path or not os.path.exists(path):
            return {}
        return {(o["test"], o["dataset"]): o["rescored"]
                for o in json.load(open(path, encoding="utf-8"))}

    base_lev = levs(args.rescored_base)
    adapted_lev = levs(args.rescored_adapted)

    print(f"base    : {base_data['model']}  (rev {str(base_data['revision_loaded'])[:12]}, "
          f"system prompt {(base_data.get('chat_template') or {}).get('system_position')}, "
          f"{'quantized' if (base_data.get('load') or {}).get('quantized') else 'PRECISION UNRECORDED'})")
    print(f"adapted : {adapted_data['model']}  (rev {str(adapted_data['revision_loaded'])[:12]}, "
          f"system prompt {(adapted_data.get('chat_template') or {}).get('system_position')}, "
          f"{'quantized' if (adapted_data.get('load') or {}).get('quantized') else 'PRECISION UNRECORDED'})")
    template_differs = ((base_data.get("chat_template") or {}).get("system_position")
                        != (adapted_data.get("chat_template") or {}).get("system_position"))
    if template_differs:
        print("\n!! the two models place the system prompt differently. Any difference "
              "below has that as a candidate cause before Russian adaptation does "
              "(AMENDMENT_3 §3).")

    order = ["header", "row", "feature", "first_token"]
    keys = sorted(set(base) | set(adapted),
                  key=lambda k: (order.index(k[0]) if k[0] in order else 9, k[1]))

    print(f"\n{'test':12s} {'dataset':22s} {'base':>12s} {'adapted':>12s} "
          f"{'lev base':>9s} {'lev adapt':>9s}  direction")
    print("-" * 92)
    wins = {"adapted higher": 0, "base higher": 0, "equal": 0, "incomparable": 0}
    for key in keys:
        b_text, b_value = summarise(base.get(key))
        a_text, a_value = summarise(adapted.get(key))
        bl = base_lev.get(key, {}).get("mean_normalized_levenshtein")
        al = adapted_lev.get(key, {}).get("mean_normalized_levenshtein")
        if b_value is None or a_value is None:
            direction = "—"
            wins["incomparable"] += 1
        elif a_value > b_value:
            direction = "adapted >"
            wins["adapted higher"] += 1
        elif a_value < b_value:
            direction = "base >"
            wins["base higher"] += 1
        else:
            direction = "="
            wins["equal"] += 1
        print(f"{key[0]:12s} {key[1]:22s} {b_text:>12s} {a_text:>12s} "
              f"{(f'{bl:.2f}' if bl is not None else ''):>9s} "
              f"{(f'{al:.2f}' if al is not None else ''):>9s}  {direction}")

    print(f"\ncomparable cells: {wins['adapted higher']} adapted higher, "
          f"{wins['base higher']} base higher, {wins['equal']} equal; "
          f"{wins['incomparable']} not comparable (a cell did not run in one of the two)")
    for label, data in (("base", base_data), ("adapted", adapted_data)):
        gate = data["gate"]
        if not gate.get("complete", True):
            missing = ", ".join(f"{c['dataset']}/{c['test']}" for c in gate["failed_cells"])
            print(f"  {label} incomplete: {gate['cells_run']}/{gate['cells_planned']} "
                  f"cells ran; missing {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
