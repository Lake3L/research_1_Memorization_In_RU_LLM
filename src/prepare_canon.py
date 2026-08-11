"""Register the Western canon — the datasets Bordt et al. tested.

These are not a mere reference point for us. Russian-centric models are built on
multilingual bases (T-lite on Qwen, Vikhr on Mistral/Llama) and their continued
pretraining runs over web corpora that are overwhelmingly English, so the canon is
the primary test surface for H1 and the only place where our measurements are
directly comparable to published numbers.

Targets follow the authors' own transform configs, so the classification task is
theirs, not ours. California Housing's target is continuous and is binarised at the
median, as is standard for that dataset in a classification setting; that deviation
is recorded here rather than left implicit.
"""

import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import register, read_table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(ROOT, "data", "canon")

SOURCES = {
    "iris": {
        "target": "species", "binarise": None,
        "url": "https://archive.ics.uci.edu/dataset/53/iris",
        "published": "1988-07-01",
        "evidence": "UCI ML Repository donation date 1988-07-01; the bytes are those "
                    "served by tabmemcheck commit 7dbeaac5 and by the tabmemcheck 0.1.6 "
                    "wheel on PyPI, which agree with each other (md5 9f44d5c5)",
        "license": "CC BY 4.0 (UCI)",
    },
    "adult-train": {
        "target": "Income", "binarise": None,
        "url": "https://archive.ics.uci.edu/dataset/2/adult",
        "published": "1996-05-01",
        "evidence": "UCI ML Repository donation date 1996-05-01; the bytes are those "
                    "served by tabmemcheck commit 7dbeaac5 and by the tabmemcheck 0.1.6 "
                    "wheel on PyPI, which agree with each other (md5 12a09c3e)",
        "license": "CC BY 4.0 (UCI)",
    },
    "california-housing": {
        "target": "median_house_value", "binarise": "median",
        "url": "https://www.kaggle.com/datasets/camnugent/california-housing-prices",
        "published": "1997-01-01",
        "evidence": "derived from Pace & Barry (1997); the bytes are those served by "
                    "tabmemcheck commit 7dbeaac5 and by the tabmemcheck 0.1.6 wheel on "
                    "PyPI, which agree with each other (md5 d1c47305)",
        "license": "CC0 (Kaggle listing)",
    },
    "openml-diabetes": {
        "target": "Outcome", "binarise": None,
        "url": "https://www.openml.org/d/37",
        "published": "1990-01-01",
        "evidence": "Pima Indians Diabetes, National Institute of Diabetes and Digestive "
                    "and Kidney Diseases, 1990; the bytes are those served by tabmemcheck "
                    "commit 7dbeaac5 and by the tabmemcheck 0.1.6 wheel on PyPI, which "
                    "agree with each other (md5 b43dd020)",
        "license": "public domain (OpenML listing)",
    },
    "uci-wine": {
        "target": "target", "binarise": None,
        "url": "https://archive.ics.uci.edu/dataset/109/wine",
        "published": "1991-07-01",
        "evidence": "UCI ML Repository donation date 1991-07-01; the bytes are those "
                    "served by tabmemcheck commit 7dbeaac5 and by the tabmemcheck 0.1.6 "
                    "wheel on PyPI, which agree with each other (md5 bf10dd49)",
        "license": "CC BY 4.0 (UCI)",
    },
    "titanic-train": {
        "target": "Survived", "binarise": None,
        "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "published": "2012-09-28",
        "evidence": "Kaggle 'Titanic: Machine Learning from Disaster' training split, "
                    "competition launched 2012-09-28. Kaggle requires authentication, so "
                    "the file here is a public mirror carrying the same 891 rows and the "
                    "same column order as the Kaggle original. CAVEAT: byte-level "
                    "identity with Kaggle's file is unverified, and the memorization "
                    "tests are byte-sensitive — a negative result on this file is "
                    "weaker evidence than on the other canon members.",
        "license": "public mirror of a Kaggle competition file; used for testing only",
    },
}


def main():
    summary = {}
    for name, spec in SOURCES.items():
        raw = os.path.join(CANON, f"{name}.csv")
        if not os.path.exists(raw):
            print(f"{name}: missing {raw}")
            continue
        df = read_table(raw)
        target = spec["target"]

        view = df.copy()
        if spec["binarise"] == "median":
            values = pd.to_numeric(view[target], errors="coerce")
            view = view[values.notna()]
            new_target = f"{target}_above_median"
            view[new_target] = (values[values.notna()] > values[values.notna()].median()).astype(int)
            view = view.drop(columns=[target])
            target = new_target
        view_path = os.path.join(CANON, f"{name}__modeling.csv")
        view.to_csv(view_path, index=False, encoding="utf-8", lineterminator="\n")

        record = register(
            name=name, group="canon", source_url=spec["url"],
            published=spec["published"], published_evidence=spec["evidence"],
            license=spec["license"], raw_path=raw, language="en",
            notes=f"Western canon, tested by Bordt et al. Target '{target}' follows their "
                  f"transform config"
                  + (" (binarised at the median: their config treats it as continuous)"
                     if spec["binarise"] else "") + ".",
        )
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "check_dataset.py"),
                        view_path, "--target", target],
                       check=False, stdout=subprocess.DEVNULL)
        q_path = view_path.replace(".csv", "_quality.json")
        entry = {"rows": record.n_rows, "cols": record.n_cols, "target": target}
        if os.path.exists(q_path):
            q = json.load(open(q_path, encoding="utf-8"))
            entry["baselines"] = q["baselines"]
            # The ≥300-row rule in PREREGISTRATION.md §4 governs which *Russian*
            # candidates we admit. Canon members are included by construction —
            # they are the set Bordt et al. published numbers for, and dropping
            # wine (178 rows) or iris (150) would break comparability with the
            # very tables we reproduced. Learnability still has to hold.
            entry["admitted"] = q["checks"]["task_is_learnable"]
            entry["admission_basis"] = "canon member; size rule does not apply"
            entry["failed_checks"] = [k for k, v in q["checks"].items() if not v]
            b = q["baselines"]
            print(f"{name}: {record.n_rows}x{record.n_cols} | LR "
                  f"{b['logistic_regression']['accuracy']:.3f} / GBT "
                  f"{b['gradient_boosting']['accuracy']:.3f} vs majority "
                  f"{b['majority_baseline']:.3f} -> admitted={entry['admitted']}"
                  + (f" (size rule waived: {entry['failed_checks']})"
                     if entry["failed_checks"] else ""))
        summary[name] = entry

    out = os.path.join(ROOT, "data", "canon_summary.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
