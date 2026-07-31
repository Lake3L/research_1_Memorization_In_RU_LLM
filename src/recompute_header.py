"""Re-score logged header-test calls without re-querying the model.

The strict prefix criterion used during the live run is harsher than the
visual judgement in the paper. This re-scores the same responses with the
criterion of metrics.header_verdict (complete rows reproduced verbatim),
reporting both numbers so the difference is visible rather than silent.
"""

import json
import os
import sys

from tabmemcheck import utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import header_verdict  # noqa: E402

DATASETS = ["iris.csv", "uci-wine.csv", "openml-diabetes.csv",
            "adult-train.csv", "california-housing.csv"]
HEADER_SYSTEM = "You are an autocomplete bot for tabular datasets."
COMPLETION_LENGTH = 500


def main(chatlog_path: str):
    texts = {d: utils.load_csv_string(d) for d in DATASETS}
    out = []
    with open(chatlog_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec["messages"]
            if not msgs or not msgs[0]["content"].startswith(HEADER_SYSTEM):
                continue
            prompt = msgs[-1]["content"]
            for name, text in texts.items():
                if text.startswith(prompt):
                    offset = len(prompt)
                    truth = text[offset:offset + COMPLETION_LENGTH]
                    v = header_verdict(truth, rec["response"])
                    v.update(dataset=name, split_offset=offset,
                             served_model=rec.get("model"))
                    out.append(v)
                    break

    by_dataset = {}
    for v in out:
        d = by_dataset.setdefault(v["dataset"], [])
        d.append(v)

    print(f"{'dataset':26s} {'strict':>8s} {'rows_recovered':>15s}  verdict")
    summary = {}
    for name in DATASETS:
        calls = by_dataset.get(name, [])
        if not calls:
            continue
        best_strict = max(c["strict_verdict"] == "pass" for c in calls)
        best_rows = max(c["rows_recovered"] for c in calls)
        verdict = "pass" if best_rows >= 1 else "fail"
        summary[name] = {"strict_pass": bool(best_strict), "max_rows_recovered": best_rows,
                         "verdict": verdict, "n_splits": len(calls)}
        print(f"{name:26s} {str(bool(best_strict)):>8s} {best_rows:>15d}  {verdict}")
    return summary


if __name__ == "__main__":
    log = sys.argv[1]
    summary = main(log)
    out_path = log.replace("chatlog_", "header_rescored_").replace(".jsonl", ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")
