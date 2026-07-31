"""Independent check of the Table 4 recomputation.

Written deliberately without reusing `recompute_table4.py`: ground truth here
comes from the dataset CSVs shipped with the authors' code, matched to each
query by its feature values, whereas the other script derives ground truth from
labels appearing in few-shot examples. Two independent paths to the same number
is the point — if they agree with each other and with the published table, the
recomputation is trustworthy.

Limited to the `original` format, where prompt values are unmodified and can be
matched against the CSV directly.
"""

import io
import json
import os
import pickle
import re
import sys
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "external", "colm-chatlogs", "colm-2024-chatlogs.zip")
DATASETS = os.path.join(ROOT, "external", "LLM-Tabular-Memorization-Checker",
                        "colm-2024-paper-code", "datasets")

# dataset -> (csv file, target column)
SPEC = {
    "iris": ("iris.csv", "species"),
    "openml-diabetes": ("openml-diabetes.csv", "Outcome"),
    "uci-wine": ("uci-wine.csv", "target"),
    "adult": ("adult-train.csv", "Income"),
    "acs-income": ("acs-income-2022.csv", None),
    "icu": ("icu.csv", None),
}

PUBLISHED = {  # Table 4, original format only
    ("gpt-3.5-0125", "iris"): 0.98, ("gpt4", "iris"): 0.99,
    ("gpt-3.5-0125", "openml-diabetes"): 0.74, ("gpt4", "openml-diabetes"): 0.74,
    ("gpt-3.5-0125", "uci-wine"): 0.88, ("gpt4", "uci-wine"): 0.96,
    ("gpt-3.5-0125", "adult"): 0.78, ("gpt4", "adult"): 0.81,
    ("gpt-3.5-0125", "acs-income"): 0.78, ("gpt4", "acs-income"): 0.78,
    ("gpt-3.5-0125", "icu"): 0.69, ("gpt4", "icu"): 0.69,
}

PAIR = re.compile(r"([\w \-\(\)\./']+?) = ([^,]*?)(?:, |,?$)")


def parse_query(text: str):
    """'IF a = 1, b = 2, THEN target =' -> ({'a': '1', 'b': '2'}, 'target')"""
    body = text.strip()
    if body.startswith("IF "):
        body = body[3:]
    m = re.search(r"THEN\s+(.+?)\s*=\s*$", body)
    target = m.group(1).strip() if m else None
    body = body[: m.start()] if m else body
    feats = {}
    for name, value in PAIR.findall(body):
        feats[name.strip()] = value.strip()
    return feats, target


def norm(v) -> str:
    s = str(v).strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f"{f:g}"
    except ValueError:
        return s


def build_index(df: pd.DataFrame, target_col: str):
    """Map a tuple of normalised non-target feature values -> set of labels."""
    feature_cols = [c for c in df.columns if c != target_col]
    index = {}
    for row in df.itertuples(index=False):
        d = dict(zip(df.columns, row))
        key = tuple(norm(d[c]) for c in feature_cols)
        index.setdefault(key, set()).add(norm(d[target_col]))
    return index, feature_cols


def evaluate(model: str, dataset: str, seed_dir: str = "0"):
    csv_name, target_col = SPEC[dataset]
    df = pd.read_csv(os.path.join(DATASETS, csv_name))
    if target_col is None:
        target_col = df.columns[-1]
    index, feature_cols = build_index(df, target_col)

    z = zipfile.ZipFile(ARCHIVE)
    prefix = f"chatlogs/{model}/{dataset}-original-{seed_dir}/"
    names = [n for n in z.namelist() if n.startswith(prefix) and n.endswith(".pkl")]

    # Logged responses are truncated to the first token or two, so a response
    # is decoded as the unique label it prefixes (see verify_table4_fewshot.py).
    vocabulary = {norm(v) for v in df[target_col].unique()}

    def decode(response: str):
        r = str(response or "").strip()
        if not r:
            return None
        if norm(r) in vocabulary:
            return norm(r)
        candidates = [lab for lab in vocabulary if lab.startswith(r)]
        return candidates[0] if len(candidates) == 1 else None

    correct = matched = ambiguous = unmatched = undecodable = 0
    for name in names:
        msgs, response = pickle.load(io.BytesIO(z.read(name)))
        feats, _ = parse_query(msgs[-1]["content"])
        key = tuple(norm(feats.get(c, "\x00missing")) for c in feature_cols)
        labels = index.get(key)
        if labels is None:
            unmatched += 1
            continue
        if len(labels) > 1:
            ambiguous += 1
            continue
        matched += 1
        prediction = decode(response)
        if prediction is None:
            undecodable += 1
            continue
        if prediction == next(iter(labels)):
            correct += 1
    return {
        "model": model, "dataset": dataset, "n_queries": len(names),
        "n_matched": matched, "n_unmatched": unmatched, "n_ambiguous": ambiguous,
        "n_undecodable_response": undecodable,
        "accuracy_on_matched": round(correct / matched, 4) if matched else None,
        "published": PUBLISHED.get((model, dataset)),
    }


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["iris", "openml-diabetes", "uci-wine", "adult", "acs-income", "icu"]
    rows = []
    for model in ("gpt-3.5-0125", "gpt4"):
        for ds in datasets:
            try:
                r = evaluate(model, ds)
            except Exception as e:
                r = {"model": model, "dataset": ds, "error": f"{type(e).__name__}: {e}"}
            rows.append(r)
            print(r)
    out = os.path.join(ROOT, "results", "table4_independent_check.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=2)
    print("wrote", out)
