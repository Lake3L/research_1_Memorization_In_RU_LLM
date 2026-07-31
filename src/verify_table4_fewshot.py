"""Second independent check, covering the transformed formats.

`verify_table4_independent.py` can only handle the `original` format, because
perturbed / task / statistical rewrite the feature values so they no longer
match the CSV. This script recovers ground truth the other way: within a cell,
every test point also appears as a few-shot example in other queries, where the
assistant turn carries its true label. That gives labels in the transformed
feature space directly.

This is the same idea as `recompute_table4.py` but implemented separately, so
agreement between the two is evidence about the idea rather than about one
implementation.
"""

import io
import json
import os
import pickle
import sys
import zipfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "external", "colm-chatlogs", "colm-2024-chatlogs.zip")


def cell(z, model, dataset, fmt, seed):
    prefix = f"chatlogs/{model}/{dataset}-{fmt}-{seed}/"
    names = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith(".pkl"))
    labels = defaultdict(set)   # query text -> labels seen as few-shot answers
    tests = []                  # (query text, model response)
    for name in names:
        msgs, response = pickle.load(io.BytesIO(z.read(name)))
        for user, assistant in zip(msgs[1:-1:2], msgs[2::2]):
            if user["role"] == "user" and assistant["role"] == "assistant":
                labels[user["content"].strip()].add(assistant["content"].strip())
        tests.append((msgs[-1]["content"].strip(), (response or "").strip()))

    # Responses in the archive are truncated to the first token or two
    # ("<=" for "<=50K", "Not" for "Not Survived"), because the authors
    # generated with a tight token limit. The authors' notebook resolves this
    # with per-dataset substring rules; the general form of the same rule is to
    # map a response to the unique label it is a prefix of.
    vocabulary = {lab for labs in labels.values() for lab in labs}

    def decode(response: str):
        if not response:
            return None
        if response in vocabulary:
            return response
        candidates = [lab for lab in vocabulary if lab.startswith(response)]
        return candidates[0] if len(candidates) == 1 else None

    correct = matched = conflict = missing = undecodable = 0
    for query, response in tests:
        truth = labels.get(query)
        if not truth:
            missing += 1
            continue
        if len(truth) > 1:
            conflict += 1
            continue
        matched += 1
        prediction = decode(response)
        if prediction is None:
            undecodable += 1
            continue
        if prediction == next(iter(truth)):
            correct += 1
    return {
        "model": model, "dataset": dataset, "format": fmt, "seed": seed,
        "n_queries": len(tests), "n_matched": matched,
        "n_missing_label": missing, "n_conflicting_label": conflict,
        "n_undecodable_response": undecodable,
        "label_vocabulary": sorted(vocabulary)[:6],
        "accuracy": round(correct / matched, 4) if matched else None,
    }


if __name__ == "__main__":
    z = zipfile.ZipFile(ARCHIVE)
    plan = [
        ("gpt4", "titanic", f, 0) for f in ("original", "perturbed", "task", "statistical")
    ] + [
        ("gpt4", "iris", f, 0) for f in ("original", "perturbed", "statistical")
    ] + [
        ("gpt4", "adult", f, 0) for f in ("original", "task", "statistical")
    ] + [
        ("gpt4", "acs-income", f, 0) for f in ("original", "task", "statistical")
    ] + [
        ("gpt-3.5-0125", "titanic", f, 0) for f in ("original", "perturbed", "task", "statistical")
    ]
    rows = [cell(z, *p) for p in plan]
    for r in rows:
        print(r)
    out = os.path.join(ROOT, "results", "table4_fewshot_check.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=2)
    print("wrote", out)
