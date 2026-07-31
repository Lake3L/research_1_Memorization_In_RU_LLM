"""Recompute Table 4 of Bordt et al., "Elephants Never Forget: Memorization and
Learning of Tabular Data in Large Language Models" (COLM 2024, arXiv:2404.06209)
directly from the authors' own published chatlogs (Zenodo 10.5281/zenodo.14644403).

This is a *positive-control* / verification step: we do not re-query any LLM.  We
re-derive the accuracy numbers of Table 4 (page 7) from the raw
``(messages, response)`` pickles that the authors released, using the authors'
own evaluation methodology as implemented in
``colm-2024-paper-code/notebooks/evaluate_tabular_experiments.ipynb``.

Methodology, as reverse-engineered from the authors' code
--------------------------------------------------------
Query construction (``run_tabular_experiments.py`` + ``tabular_queries.py``):

* ``df = tabmemcheck.datasets.load_dataset(csv, yaml, transform, seed=seed)``;
  the target is the *last* column, all values are cast to ``str``.
* If the dataset has more than 1300 rows, a ``train_test_split(test_size=0.2,
  random_state=42)`` is performed and only ``X_train`` is used (this shuffles
  the rows; the permutation depends only on ``n_rows`` and ``random_state=42``,
  therefore it is identical for every transform and every seed of a dataset).
* ``statutils.loo_eval`` then loops ``idx = 0 .. min(n_rows, 1000) - 1``.  For
  each ``idx`` it builds a 20-shot chat prompt from the *other* rows and asks the
  model to complete ``"IF f1 = v1, ..., THEN <target> ="``.
* ``tabmemcheck.llm.log`` writes the i-th query of an experiment to
  ``chatlogs/<exp>/<exp>-<i>.pkl``.  **Hence the file index i is exactly the row
  index idx into ``X_train``** -- this is the key that lets us recover the
  ground truth.  We verify it explicitly (see ``verify`` counts in the output).

Evaluation (``evaluate_tabular_experiments.ipynb``):

* ``y_true = y_train[:1000]`` (from the dataframe), ``y_pred = responses`` read
  in file-index order by ``tabmemcheck.read_chatlog``.
* Both are mapped to integer class ids with *hand-written, per-dataset,
  per-transform string rules* (e.g. ``0 if "Less" in r else 1`` for Adult/task,
  ``int(x) if x.isdigit() else 0`` for the statistical transform).  Those rules
  are reproduced verbatim in ``TRUTH_RULES`` / ``PRED_RULES`` below.
* ``accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)``.

Ground-truth recovery
---------------------
Two independent sources are used and cross-checked:

1. **CSV reconstruction** (primary where possible).  For the 7 datasets whose
   CSV ships with the authors' code (adult, acs-income, acs-travel, icu, iris,
   uci-wine, openml-diabetes) we re-run ``load_dataset`` with the same
   transform/seed and take ``y_train[:1000]``.  Correctness of the row alignment
   is *verified* by re-rendering the prompt for every test index with the
   authors' ``format_data_point`` and comparing it byte-for-byte with the last
   user message stored in the pickle.
2. **Log reconstruction** (used for titanic, spaceship-titanic and fico, whose
   CSVs are not redistributed for licensing reasons).  Every conversation
   contains 20 labelled few-shot rows of the *same* dataset version, i.e. a
   ``prompt-string -> label`` dictionary.  Pooling those over all conversations
   (and, via the index correspondence described above, over all transforms,
   seeds and models of the same dataset) recovers the label of essentially every
   test row.  For the ``statistical`` transform the target coding is a random
   permutation of the class ids, which we identify from the overlap with the
   canonical (original-transform) labels.

Everything that could not be recovered is reported as ``n_missing`` and excluded
from the accuracy (never guessed).

Outputs
-------
``results/table4_recomputed_from_authors_logs.json``  (full detail, per seed)
``results/table4_recomputed_from_authors_logs.csv``   (aggregated table + deltas)

Run with the project venv::

    .venv/Scripts/python.exe src/recompute_table4.py
"""

from __future__ import annotations

import csv as csvmod
import json
import os
import pickle
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT_ROOT / "external" / "colm-chatlogs" / "colm-2024-chatlogs.zip"
CODE_DIR = (
    PROJECT_ROOT
    / "external"
    / "LLM-Tabular-Memorization-Checker"
    / "colm-2024-paper-code"
)
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_STEM = "table4_recomputed_from_authors_logs"

sys.path.insert(0, str(CODE_DIR))
import tabmemcheck  # noqa: E402  (authors' package, editable install)
from sklearn.model_selection import train_test_split  # noqa: E402
from tabular_queries import format_data_point  # noqa: E402  (authors' code)

tabmemcheck.config.csv_max_rows = 10**9  # the ACS files have >100k rows

# --------------------------------------------------------------------------- #
# experiment grid
# --------------------------------------------------------------------------- #

MODELS = ["gpt4", "gpt-3.5-0125"]
FORMATS = ["original", "perturbed", "task", "statistical"]

# dataset -> (csv file, yaml transform config) relative to CODE_DIR; None = CSV
# not redistributed by the authors -> ground truth is recovered from the logs.
DATASETS = {
    "titanic": None,
    "adult": ("datasets/adult-train.csv", "config/transform/adult.yaml"),
    "openml-diabetes": (
        "datasets/openml-diabetes.csv",
        "config/transform/openml-diabetes.yaml",
    ),
    "uci-wine": ("datasets/uci-wine.csv", "config/transform/uci-wine.yaml"),
    "iris": ("datasets/iris.csv", "config/transform/iris.yaml"),
    "spaceship-titanic": None,
    "acs-income": ("datasets/acs-income-2022.csv", "config/transform/acs-income.yaml"),
    "icu": ("datasets/icu.csv", "config/transform/icu.yaml"),
    "fico": None,
    "acs-travel": ("datasets/acs-travel-2022.csv", "config/transform/acs-travel.yaml"),
}

PANEL_A = ["titanic", "adult", "openml-diabetes", "uci-wine", "iris"]  # memorized
PANEL_B = ["spaceship-titanic", "acs-income", "icu", "fico", "acs-travel"]  # novel

# Table 4 of the paper (page 7), as provided in the reproduction protocol.
# {(dataset, format): {"gpt-3.5-0125": x, "gpt4": y}};  cells absent from the
# protocol are simply missing here and are reported as "n/a".
PUBLISHED = {
    ("titanic", "original"): {"gpt-3.5-0125": 0.81, "gpt4": 0.96},
    ("titanic", "perturbed"): {"gpt-3.5-0125": 0.78, "gpt4": 0.82},
    ("titanic", "task"): {"gpt-3.5-0125": 0.77, "gpt4": 0.80},
    ("titanic", "statistical"): {"gpt-3.5-0125": 0.61, "gpt4": 0.65},
    ("adult", "original"): {"gpt-3.5-0125": 0.78, "gpt4": 0.81},
    ("adult", "task"): {"gpt-3.5-0125": 0.75, "gpt4": 0.79},
    ("adult", "statistical"): {"gpt-3.5-0125": 0.70, "gpt4": 0.63},
    ("openml-diabetes", "original"): {"gpt-3.5-0125": 0.74, "gpt4": 0.74},
    ("openml-diabetes", "statistical"): {"gpt-3.5-0125": 0.68, "gpt4": 0.62},
    ("uci-wine", "original"): {"gpt-3.5-0125": 0.88, "gpt4": 0.96},
    ("uci-wine", "task"): {"gpt-3.5-0125": 0.87, "gpt4": 0.87},
    ("iris", "original"): {"gpt-3.5-0125": 0.98, "gpt4": 0.99},
    ("iris", "perturbed"): {"gpt-3.5-0125": 0.95, "gpt4": 0.95},
    ("iris", "statistical"): {"gpt-3.5-0125": 0.87, "gpt4": 0.92},
    ("spaceship-titanic", "original"): {"gpt-3.5-0125": 0.58, "gpt4": 0.67},
    ("spaceship-titanic", "statistical"): {"gpt-3.5-0125": 0.63, "gpt4": 0.66},
    ("acs-income", "original"): {"gpt-3.5-0125": 0.78, "gpt4": 0.78},
    ("acs-income", "task"): {"gpt-3.5-0125": 0.77, "gpt4": 0.77},
    ("acs-income", "statistical"): {"gpt-3.5-0125": 0.59, "gpt4": 0.57},
    ("icu", "original"): {"gpt-3.5-0125": 0.69, "gpt4": 0.69},
    ("fico", "original"): {"gpt-3.5-0125": 0.58, "gpt4": 0.67},
    ("acs-travel", "original"): {"gpt-3.5-0125": 0.54, "gpt4": 0.62},
}

# --------------------------------------------------------------------------- #
# label parsing -- verbatim from evaluate_tabular_experiments.ipynb
# --------------------------------------------------------------------------- #


def _digit(s):
    """The notebook's ``int(x) if x.isdigit() else 0``."""
    s = "" if s is None else str(s).strip()
    return int(s) if s.isdigit() else None  # None -> counted as unparsed


def _contains(needle, if_true=0, if_false=1):
    def fn(s):
        if s is None:
            return if_true  # notebook: ``0 if r is None or ... else 1``
        return if_true if needle in s else if_false

    return fn


def _iris(s):
    if s is None:
        return None
    return 0 if "etosa" in s else 1 if "ersicolor" in s else 2


def _wine_task(s):
    if s is None:
        return None
    return 1 if "1" in s else 2 if "2" in s else 3


def _diabetes_task(s):
    if s is None:
        return None
    return 1 if "Pos" in s else 0


# ground-truth rules (applied to the label string as it appears in the
# dataframe / in the assistant messages of the logs)
TRUTH_RULES = {
    ("acs-income", "original"): _contains("Less"),
    ("acs-income", "perturbed"): _contains("Less"),
    ("acs-income", "task"): _contains("Less"),
    ("acs-travel", "original"): _contains("Less"),
    ("acs-travel", "perturbed"): _contains("Less"),
    ("acs-travel", "task"): _contains("Shorter"),
    ("adult", "original"): _contains("<="),
    ("adult", "perturbed"): _contains("<="),
    ("adult", "task"): _contains("Less"),
    ("fico", "original"): _contains("Bad"),
    ("fico", "perturbed"): _contains("Bad"),
    ("fico", "task"): _contains("Def"),
    ("titanic", "original"): _digit,
    ("titanic", "perturbed"): _digit,
    ("titanic", "task"): _contains("Not"),
    ("spaceship-titanic", "original"): _contains("F"),
    ("spaceship-titanic", "perturbed"): _contains("F"),
    ("spaceship-titanic", "task"): _contains("No"),
    ("iris", "original"): _iris,
    ("iris", "perturbed"): _iris,
    ("iris", "task"): _iris,
    ("uci-wine", "original"): _digit,
    ("uci-wine", "perturbed"): _digit,
    ("uci-wine", "task"): _wine_task,
    ("openml-diabetes", "original"): _digit,
    ("openml-diabetes", "perturbed"): _digit,
    ("openml-diabetes", "task"): _diabetes_task,
    ("icu", "original"): _digit,
    ("icu", "perturbed"): _digit,
    ("icu", "task"): _contains("IC"),
}

# prediction rules; identical to the truth rules except where the notebook uses
# a shorter (more permissive) substring for the model response.
PRED_RULES = dict(TRUTH_RULES)
PRED_RULES[("acs-travel", "original")] = _contains("L")
PRED_RULES[("acs-travel", "perturbed")] = _contains("L")
PRED_RULES[("acs-travel", "task")] = _contains("Short")

for _ds in DATASETS:  # the statistical transform always uses plain digits
    TRUTH_RULES[(_ds, "statistical")] = _digit
    PRED_RULES[(_ds, "statistical")] = _digit


# --------------------------------------------------------------------------- #
# reading the chatlog archive
# --------------------------------------------------------------------------- #


def scan_archive(zf: zipfile.ZipFile):
    """Return {(model, exp_name): n_files} for every experiment in the archive."""
    counts = Counter()
    for name in zf.namelist():
        if not name.endswith(".pkl"):
            continue
        parts = name.split("/")
        if len(parts) != 4:
            continue
        counts[(parts[1], parts[2])] += 1
    return counts


def read_experiment(zf, model, exp, n, want_shots=False):
    """Read one experiment.

    Returns ``(test_prompts, responses, shots)`` where ``shots`` maps the
    few-shot prompt string to the set of labels observed for it.
    """
    prompts, responses = [], []
    shots = defaultdict(set) if want_shots else None
    for i in range(n):
        with zf.open(f"chatlogs/{model}/{exp}/{exp}-{i}.pkl") as fh:
            messages, response = pickle.load(fh)
        prompts.append(messages[-1]["content"])
        responses.append(response)
        if want_shots:
            body = messages[1:-1]
            for j in range(0, len(body) - 1, 2):
                shots[body[j]["content"]].add(body[j + 1]["content"])
    return prompts, responses, shots


# --------------------------------------------------------------------------- #
# ground truth: CSV reconstruction
# --------------------------------------------------------------------------- #


def csv_ground_truth(dataset, fmt, seed, n):
    """Re-create ``y_train[:n]`` and the corresponding prompt strings."""
    csv_file, yaml_file = DATASETS[dataset]
    df = tabmemcheck.datasets.load_dataset(
        str(CODE_DIR / csv_file), str(CODE_DIR / yaml_file), fmt, seed=seed
    )
    feature_names, target_name = df.columns.tolist()[:-1], df.columns.tolist()[-1]
    X = df[feature_names].astype("str").values
    y = df[target_name].astype("str").values
    if X.shape[0] > 1300:
        X, _, y, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    prompts = [
        format_data_point(X[i], feature_names, add_if_then=True) + f" {target_name} ="
        for i in range(min(n, X.shape[0]))
    ]
    return list(y[:n]), prompts


# --------------------------------------------------------------------------- #
# ground truth: log reconstruction (titanic / spaceship-titanic / fico)
# --------------------------------------------------------------------------- #


def log_ground_truth(zf, counts, dataset, n):
    """Recover the label of every test index of ``dataset`` from the chatlogs.

    Returns ``{(fmt, seed): {index: label_string}}``.  The few-shot examples of
    every conversation form a ``prompt -> label`` dictionary for the dataset
    version at hand; prompts that occur with more than one label (duplicate rows
    with different targets) are treated as unrecoverable.
    """
    per_exp = {}  # (fmt, seed) -> {idx: label}
    all_models = sorted({m for (m, _e) in counts})
    for fmt in FORMATS:
        for seed in range(10):
            exp = f"{dataset}-{fmt}-{seed}"
            models = [m for m in all_models if (m, exp) in counts]
            if not models:
                continue
            n_exp = counts[(models[0], exp)]
            shots = defaultdict(set)
            prompts = None
            for model in models:
                p, _r, s = read_experiment(zf, model, exp, n_exp, want_shots=True)
                for k, v in s.items():
                    shots[k] |= v
                if prompts is None:
                    prompts = p
            labels = {}
            for i, p in enumerate(prompts):
                v = shots.get(p)
                if v is not None and len(v) == 1:
                    labels[i] = next(iter(v))
            per_exp[(fmt, seed)] = labels
            if n_exp != n:  # pragma: no cover - sanity
                raise RuntimeError(f"{exp}: {n_exp} files, expected {n}")
    return per_exp


def build_log_gt(zf, counts, dataset, n):
    """Canonical class per test index + per-experiment ground truth."""
    per_exp = log_ground_truth(zf, counts, dataset, n)

    # 1. canonical class ids from the non-statistical transforms (their class
    #    spaces are aligned by construction of the notebook's parsing rules)
    votes = defaultdict(Counter)
    for (fmt, _seed), labels in per_exp.items():
        if fmt == "statistical":
            continue
        rule = TRUTH_RULES[(dataset, fmt)]
        for i, lab in labels.items():
            c = rule(lab)
            if c is not None:
                votes[i][c] += 1
    canonical, canonical_conflicts = {}, 0
    for i, cnt in votes.items():
        if len(cnt) > 1:
            canonical_conflicts += 1
        canonical[i] = cnt.most_common(1)[0][0]

    # 2. per-experiment ground truth (class ids)
    gt = {}
    stat_maps = {}
    for (fmt, seed), labels in per_exp.items():
        rule = TRUTH_RULES[(dataset, fmt)]
        observed = {}
        for i, lab in labels.items():
            c = rule(lab)
            if c is not None:
                observed[i] = c
        if fmt != "statistical":
            merged = dict(canonical)
            merged.update(observed)  # direct observation wins
            gt[(fmt, seed)] = merged
        else:
            # the statistical transform re-codes the target with a random
            # permutation; identify it from the overlap with `canonical`
            mapping = defaultdict(Counter)
            for i, c in observed.items():
                if i in canonical:
                    mapping[canonical[i]][c] += 1
            perm = {k: v.most_common(1)[0][0] for k, v in mapping.items()}
            stat_maps[seed] = {
                "map": {str(k): v for k, v in perm.items()},
                "support": {str(k): sum(v.values()) for k, v in mapping.items()},
                "purity": {
                    str(k): v.most_common(1)[0][1] / sum(v.values())
                    for k, v in mapping.items()
                },
            }
            merged = {
                i: perm[c] for i, c in canonical.items() if c in perm
            }
            merged.update(observed)
            gt[(fmt, seed)] = merged
    return gt, len(canonical), canonical_conflicts, stat_maps


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    zf = zipfile.ZipFile(ZIP_PATH)
    counts = scan_archive(zf)

    rows = []  # per (model, dataset, format, seed)
    notes = {}

    # --- ground truth for the three datasets without a public CSV ---------- #
    log_gt = {}
    for dataset in DATASETS:
        if DATASETS[dataset] is not None:
            continue
        n = counts[("gpt4", f"{dataset}-original-0")]
        gt, n_canon, conflicts, stat_maps = build_log_gt(zf, counts, dataset, n)
        log_gt[dataset] = gt
        notes[dataset] = {
            "ground_truth_source": "chatlogs (few-shot examples)",
            "n_test_points": n,
            "n_indices_recovered": n_canon,
            "canonical_label_conflicts": conflicts,
            "statistical_class_permutation": stat_maps,
        }
        print(
            f"[log-gt] {dataset}: recovered {n_canon}/{n} test labels "
            f"({conflicts} conflicting)"
        )

    # --- evaluate every experiment ----------------------------------------- #
    for model in MODELS:
        for dataset in DATASETS:
            for fmt in FORMATS:
                for seed in range(10):
                    exp = f"{dataset}-{fmt}-{seed}"
                    if (model, exp) not in counts:
                        continue
                    n = counts[(model, exp)]
                    prompts, responses, _ = read_experiment(zf, model, exp, n)

                    truth_rule = TRUTH_RULES[(dataset, fmt)]
                    pred_rule = PRED_RULES[(dataset, fmt)]

                    n_prompt_mismatch = None
                    if DATASETS[dataset] is not None:
                        y_str, ref_prompts = csv_ground_truth(dataset, fmt, seed, n)
                        n_prompt_mismatch = sum(
                            1 for a, b in zip(prompts, ref_prompts) if a != b
                        )
                        gt = {}
                        for i, s in enumerate(y_str):
                            c = truth_rule(s)
                            if c is not None:
                                gt[i] = c
                    else:
                        gt = log_gt[dataset][(fmt, seed)]

                    n_correct = n_eval = n_unparsed = 0
                    for i, resp in enumerate(responses):
                        if i not in gt:
                            continue
                        p = pred_rule(resp)
                        n_eval += 1
                        if p is None:
                            n_unparsed += 1
                            continue  # unparseable response counts as wrong
                        if p == gt[i]:
                            n_correct += 1

                    rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "format": fmt,
                            "seed": seed,
                            "n_queries": n,
                            "n_evaluated": n_eval,
                            "n_missing_ground_truth": n - n_eval,
                            "n_unparsed_responses": n_unparsed,
                            "accuracy": n_correct / n_eval if n_eval else None,
                            "n_prompt_mismatch_vs_csv": n_prompt_mismatch,
                        }
                    )
                    print(
                        f"{model:14s} {exp:32s} n={n_eval:5d}/{n:<5d} "
                        f"acc={rows[-1]['accuracy']:.4f} "
                        f"unparsed={n_unparsed} "
                        f"prompt_mismatch={n_prompt_mismatch}"
                    )

    # --- aggregate over seeds ---------------------------------------------- #
    agg = []
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["model"], r["dataset"], r["format"])].append(r)
    for (model, dataset, fmt), rs in by_cell.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        accs = [r["accuracy"] for r in rs]
        pub = PUBLISHED.get((dataset, fmt), {}).get(model)
        mean_acc = float(np.mean(accs))
        agg.append(
            {
                "model": model,
                "dataset": dataset,
                "format": fmt,
                "panel": "A_memorized" if dataset in PANEL_A else "B_novel",
                "seeds": [r["seed"] for r in rs],
                "n_seeds": len(rs),
                "accuracy_mean_over_seeds": round(mean_acc, 4),
                "accuracy_seed0": round(accs[0], 4),
                "accuracy_per_seed": [round(a, 4) for a in accs],
                "n_evaluated_seed0": rs[0]["n_evaluated"],
                "n_missing_ground_truth_seed0": rs[0]["n_missing_ground_truth"],
                "n_prompt_mismatch_vs_csv_seed0": rs[0]["n_prompt_mismatch_vs_csv"],
                "published_table4": pub,
                "delta_vs_published": (
                    round(mean_acc - pub, 4) if pub is not None else None
                ),
            }
        )

    order = {d: i for i, d in enumerate(PANEL_A + PANEL_B)}
    agg.sort(
        key=lambda r: (r["model"], order[r["dataset"]], FORMATS.index(r["format"]))
    )

    # --- the paper's headline effect --------------------------------------- #
    effect = {}
    for model in MODELS:
        acc = {
            (r["dataset"], r["format"]): r["accuracy_mean_over_seeds"]
            for r in agg
            if r["model"] == model
        }
        for panel, dss in (("A_memorized", PANEL_A), ("B_novel", PANEL_B)):
            drops_p = [acc[(d, "perturbed")] - acc[(d, "original")] for d in dss]
            drops_t = [acc[(d, "task")] - acc[(d, "original")] for d in dss]
            drops_s = [acc[(d, "statistical")] - acc[(d, "original")] for d in dss]
            effect[f"{model}|{panel}"] = {
                "mean_original": round(float(np.mean([acc[(d, "original")] for d in dss])), 4),
                "mean_delta_perturbed_minus_original": round(float(np.mean(drops_p)), 4),
                "mean_delta_task_minus_original": round(float(np.mean(drops_t)), 4),
                "mean_delta_statistical_minus_original": round(float(np.mean(drops_s)), 4),
                "mean_delta_perturbed_and_task": round(
                    float(np.mean(drops_p + drops_t)), 4
                ),
            }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Table 4 of Bordt et al. (COLM 2024) recomputed from the authors' "
            "published chatlogs using the authors' own evaluation rules."
        ),
        "source_archive": str(ZIP_PATH),
        "authors_code": str(CODE_DIR),
        "models": MODELS,
        "ground_truth_notes": notes,
        "aggregated": agg,
        "per_seed": rows,
        "headline_effect": effect,
    }
    with open(RESULTS_DIR / f"{OUT_STEM}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    fields = [
        "model",
        "dataset",
        "format",
        "panel",
        "n_seeds",
        "accuracy_mean_over_seeds",
        "accuracy_seed0",
        "n_evaluated_seed0",
        "n_missing_ground_truth_seed0",
        "published_table4",
        "delta_vs_published",
    ]
    with open(RESULTS_DIR / f"{OUT_STEM}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csvmod.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in agg:
            w.writerow(r)

    print(f"\nwrote {RESULTS_DIR / (OUT_STEM + '.json')}")
    print(f"wrote {RESULTS_DIR / (OUT_STEM + '.csv')}")
    for k, v in effect.items():
        print(k, v)


if __name__ == "__main__":
    main()
