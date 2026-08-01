"""Prepare the Russian pre-cutoff datasets: fetch at a pinned commit, define the
classification task, check quality, register.

These sources are reference tables, not ML benchmarks, so none ships with a target
column. The memorization tests do not need one — they ask whether the model can
reproduce rows verbatim. H3 does, so each dataset gets a task defined here, in the
same spirit as the fresh control (a threshold on a quantity the table already
contains, or an existing binary attribute).

Two artefacts per dataset, kept distinct on purpose:
- the *published file*, byte for byte, which is what the memorization tests run on
  and what the model could have seen;
- a *modeling view*, which drops row identifiers and constant columns and adds the
  target. This is what the few-shot experiments use.

GitHub is pinned by commit SHA rather than branch: `hflabs/fms-unit` still receives
updates in 2026, so `master` would silently stop being pre-cutoff data.
"""

import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import register, read_table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "ru_pre_cutoff")

SOURCES = {
    "hflabs_city": {
        "repo": "hflabs/city", "path": "city.csv",
        "url": "https://github.com/hflabs/city",
        "license": "CC BY-SA 4.0 (stated in repository README)",
        "target": "Население выше медианы",
        "target_from": ("population", "median"),
        "drop": ["population"],
        "note": "1117 Russian cities, DaData/HFLabs reference. 206 stars, 121 forks — "
                "the most widely reused of the candidates, so the strongest prior for "
                "having entered pretraining corpora.",
    },
    "govdomains": {
        "repo": "infoculture/govdomains", "path": "refined/feddomains.csv",
        "url": "https://github.com/infoculture/govdomains",
        "license": "Other (LICENSE.txt in repository)",
        "target": "Поддержка HTTPS",
        "target_from": ("HTTPS Support", "binary_ru"),
        "drop": ["HTTPS Support"],
        "note": "Russian federal government domains, Infoculture.",
    },
    "mos_zemelnye_uchastki": {
        "repo": "infoculture/mosopendata",
        "path": "old/thedata/483_zemelnye_uchastki_v_sobstvennosti_goroda_moskvy.csv",
        "url": "https://github.com/infoculture/mosopendata",
        "license": "not stated; mirror of data.mos.ru publications",
        "target": "Аренда",
        "target_from": ("Аренда", "as_is"),
        "drop": ["№ строки", "Идентификатор", "Тип собственника"],
        "note": "Moscow city land plots. The only verified source with Cyrillic column "
                "names as well as values. Tab-separated despite the .csv extension. "
                "Repository frozen since 2014.",
    },
    "mos_torgovye_obekty": {
        "repo": "infoculture/mosopendata",
        "path": "old/thedata/486_nestatsionarnye_torgovye_obekty.csv",
        "url": "https://github.com/infoculture/mosopendata",
        "license": "not stated; mirror of data.mos.ru publications",
        "target": "Модульный объект",
        "target_from": ("Вид нестационарного объекта", "equals:Модульный объект"),
        "drop": ["№ строки", "Номер свидетельства", "Дата свидетельства",
                 "Вид нестационарного объекта"],
        "note": "Moscow non-stationary retail objects, 10212 rows. Cyrillic column "
                "names and values; tab-separated despite the .csv extension.",
    },
    "russian_retail": {
        "repo": None, "path": None,
        "url": "https://www.kaggle.com/datasets/pavelkunitsyn/russian-retail",
        "license": "Other (specified in description); unclear — flagged in the amendment",
        "target": "Средний ценовой сегмент",
        "target_from": ("price_category", "equals:Средний"),
        "drop": ["price_category", "description"],
        "note": "Russian retail brands from Kaggle, 2737 rows. English headers, Cyrillic "
                "values. The free-text 'description' column is dropped from the modeling "
                "view (it is prose, not a tabular feature) but stays in the published "
                "file, where it is the highest-entropy field for the completion tests.",
        "published": "2021-08-11",
        "published_evidence": "Kaggle API lastUpdated=2021-08-11T18:49:36Z; file mtime "
                              "inside the downloaded archive 2021-08-11 18:49:38",
    },
}


def pinned_sha(repo: str, path: str) -> tuple:
    """Commit that last touched this file — the date evidence for 'pre-cutoff'."""
    import urllib.request
    url = f"https://api.github.com/repos/{repo}/commits?path={path}&per_page=1"
    with urllib.request.urlopen(url, timeout=60) as r:
        commits = json.load(r)
    if not commits:
        return None, None
    c = commits[0]
    return c["sha"], c["commit"]["committer"]["date"]


def build_modeling_view(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    source_col, mode = spec["target_from"]
    view = df.copy()
    if mode == "median":
        values = pd.to_numeric(view[source_col], errors="coerce")
        view = view[values.notna()]
        values = values[values.notna()]
        target = (values > values.median()).astype(int)
    elif mode.startswith("equals:"):
        target = (view[source_col].astype(str).str.strip() == mode.split(":", 1)[1]).astype(int)
    elif mode == "binary_ru":
        target = (view[source_col].astype(str).str.strip() == "Да").astype(int)
    else:
        target = (view[source_col].astype(str).str.strip() == "Да").astype(int)
    view = view.drop(columns=[c for c in spec["drop"] if c in view.columns])
    # constant columns carry no information and confuse the baselines
    constant = [c for c in view.columns if view[c].nunique(dropna=False) <= 1]
    view = view.drop(columns=constant)
    view[spec["target"]] = target.values
    return view.reset_index(drop=True), constant


def main():
    summary = {}
    for name, spec in SOURCES.items():
        raw_path = os.path.join(RAW, f"{name}.csv")
        if not os.path.exists(raw_path):
            print(f"{name}: raw file missing, download it first")
            continue
        if spec["repo"]:
            sha, date = pinned_sha(spec["repo"], spec["path"])
            source_url = f"https://raw.githubusercontent.com/{spec['repo']}/{sha}/{spec['path']}"
            evidence = (f"last commit touching this file: {sha} dated {date} "
                        f"(GitHub API, repo {spec['repo']}); pinned by SHA so the "
                        f"artefact cannot change under us")
        else:  # not on GitHub: provenance comes from the platform's own metadata
            sha, date = None, spec["published"]
            source_url = spec["url"]
            evidence = spec["published_evidence"]
        df = read_table(raw_path)
        view, constant = build_modeling_view(df, spec)

        view_path = os.path.join(RAW, f"{name}__modeling.csv")
        view.to_csv(view_path, index=False, encoding="utf-8", lineterminator="\n")

        record = register(
            name=name, group="ru_pre_cutoff", source_url=source_url,
            published=(date or "")[:10], published_evidence=evidence,
            license=spec["license"], raw_path=raw_path, language="ru",
            notes=spec["note"] + f" Modeling view: target '{spec['target']}', "
                                 f"dropped {spec['drop']}"
                                 + (f", constant columns {constant}" if constant else ""),
        )
        summary[name] = {"rows": record.n_rows, "cols": record.n_cols,
                         "sha": sha, "commit_date": date,
                         "modeling_view": os.path.relpath(view_path, ROOT),
                         "target": spec["target"],
                         "cyrillic_headers": record.diagnostics["cyrillic_in_headers"],
                         "entropy_feature": record.diagnostics["most_unique_feature"]}
        print(f"{name}: {record.n_rows}x{record.n_cols}, pinned {sha[:8]} ({date[:10]})")

        subprocess.run([sys.executable, os.path.join(ROOT, "src", "check_dataset.py"),
                        view_path, "--target", spec["target"]],
                       check=False, stdout=subprocess.DEVNULL)
        quality_path = view_path.replace(".csv", "_quality.json")
        if os.path.exists(quality_path):
            q = json.load(open(quality_path, encoding="utf-8"))
            summary[name]["baselines"] = q["baselines"]
            summary[name]["admitted"] = q["admitted"]
            summary[name]["failed_checks"] = [k for k, v in q["checks"].items() if not v]
            print(f"  LR {q['baselines']['logistic_regression']['accuracy']:.3f} / "
                  f"GBT {q['baselines']['gradient_boosting']['accuracy']:.3f} vs "
                  f"majority {q['baselines']['majority_baseline']:.3f} -> "
                  f"admitted={q['admitted']} {summary[name]['failed_checks'] or ''}")

    out = os.path.join(ROOT, "data", "ru_pre_cutoff_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
