"""Collect the fresh control dataset: Russian job vacancies posted after the models' cutoffs.

This is the zero line the whole study rests on (working plan §2.4): a Russian-language
table that no tested model can have memorized, because the records postdate every
model and the file itself is assembled by us. Memorization tests must score zero here;
if they do not, the method is broken and no positive result elsewhere means anything.

Design mirrors Bordt et al.'s ACS Income, which is a modern rebuild of the classic
Adult task: predict whether pay exceeds a threshold from job and demographic-style
attributes. Here the source is trudvsem.ru ("Работа России"), the Russian federal
employment service, whose open API is published for exactly this purpose and needs no
key.

Leakage is the trap. The raw records carry the salary in free text — 82 of 100 sampled
`duty` fields read like "Средний доход 80000 – 100000 руб. в месяц", and `salary` is the
figure itself. Those fields are dropped, and a guard re-checks every retained cell
against the target value before anything is written.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "http://opendata.trudvsem.ru/api/v1/vacancies/region/{region}"

# Fields that carry the target in plain text, or identify the row rather than
# describing it. Dropped before anything else happens.
LEAKING = {"salary", "salary_min", "salary_max", "duty", "vac_url", "id",
           "contact_list", "contact_person", "date_modify", "source"}

REGIONS = {
    "7700000000000": "Москва",
    "7800000000000": "Санкт-Петербург",
    "5000000000000": "Московская область",
    "6600000000000": "Свердловская область",
    "5400000000000": "Новосибирская область",
    "1600000000000": "Республика Татарстан",
    "2300000000000": "Краснодарский край",
    "6600000000001": "—",  # placeholder ignored if the API rejects it
}

COLUMNS_RU = {
    "region": "Регион",
    "specialisation": "Специализация",
    "job_name": "Должность",
    "education": "Образование",
    "experience": "Опыт работы (лет)",
    "schedule": "График работы",
    "work_places": "Количество мест",
    "hr_agency": "Кадровое агентство",
    "quota": "Квотируемое место",
    "for_disabled": "Специальное рабочее место",
    "social_protected": "Социальная защита",
    "training_days": "Дни обучения",
    "company_name": "Работодатель",
    "creation_date": "Дата публикации",
}


def fetch_region(region: str, pages: int, pause: float = 0.5) -> list:
    out = []
    for page in range(pages):
        r = requests.get(API.format(region=region),
                         params={"offset": page, "limit": 100}, timeout=60)
        r.raise_for_status()
        chunk = (r.json().get("results") or {}).get("vacancies") or []
        if not chunk:
            break
        out.extend(v["vacancy"] for v in chunk)
        time.sleep(pause)
    return out


def to_row(v: dict) -> dict:
    company = v.get("company") or {}
    req = v.get("requirement") or {}
    cat = v.get("category") or {}
    wp = v.get("workPlaceType") or {}
    return {
        "region": (v.get("region") or {}).get("name"),
        "specialisation": cat.get("specialisation"),
        "job_name": v.get("job-name"),
        "education": req.get("education"),
        "experience": req.get("experience"),
        "schedule": v.get("schedule"),
        "work_places": v.get("work_places"),
        "hr_agency": company.get("hr-agency"),
        "quota": wp.get("workPlaceQuota"),
        "for_disabled": wp.get("workPlaceSpecial"),
        "social_protected": v.get("social_protected"),
        "training_days": v.get("trainingDays"),
        "company_name": company.get("name"),
        "creation_date": v.get("creation-date"),
        "_salary_min": v.get("salary_min"),   # target source, dropped after use
    }


def assert_no_leakage(df: pd.DataFrame, salaries: pd.Series) -> None:
    """No retained cell may contain the salary figure that defines the target."""
    offenders = []
    for column in df.columns:
        values = df[column].astype(str)
        for cell, salary in zip(values, salaries):
            if salary and salary > 0 and re.search(rf"\b{int(salary)}\b", cell):
                offenders.append((column, cell[:60], salary))
                break
    if offenders:
        raise AssertionError(f"target leaks into features: {offenders[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-06-01",
                    help="keep vacancies created on or after this date")
    ap.add_argument("--pages-per-region", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "fresh_control",
                                                  "trudvsem_vacancies_2026.csv"))
    args = ap.parse_args()

    raw = []
    for code, name in REGIONS.items():
        try:
            got = fetch_region(code, args.pages_per_region)
        except Exception as e:
            print(f"  {name}: skipped ({type(e).__name__})")
            continue
        print(f"  {name}: {len(got)} vacancies")
        raw.extend(got)

    df = pd.DataFrame(to_row(v) for v in raw)
    # the API returns "" rather than null for absent values, which would otherwise
    # survive the sparse-column check below and enter the table as a constant
    df = df.replace(r"^\s*$", np.nan, regex=True)
    before = len(df)
    df = df[df["creation_date"] >= args.since]
    df = df[df["_salary_min"].notna() & (df["_salary_min"] > 0)]
    df = df.drop_duplicates()
    print(f"kept {len(df)} of {before} rows (created >= {args.since}, salary present)")
    if len(df) < 300:
        raise SystemExit(f"only {len(df)} rows; the preregistered minimum is 300")

    salaries = df.pop("_salary_min")
    threshold = float(salaries.median())
    assert_no_leakage(df, salaries)

    df["target"] = (salaries > threshold).astype(int).values

    # Housekeeping that has to happen after the target exists, or it undoes itself:
    # identical postings differing only in salary become duplicates once the salary
    # column is gone, and duplicate rows would inflate the base rate that the
    # row-completion test is measured against.
    dropped_sparse = [c for c in df.columns
                      if c != "target" and df[c].isna().mean() > 0.5]
    df = df.drop(columns=dropped_sparse)
    date_span = (df["creation_date"].min(), df["creation_date"].max())
    df = df.drop(columns=["creation_date"])  # provenance, not a job attribute
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"dropped {before_dedup - len(df)} duplicate rows; "
          f"dropped sparse columns: {dropped_sparse or 'none'}")

    df = df.rename(columns={**COLUMNS_RU, "target": "Зарплата выше медианы"})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")

    meta = {
        "source": "trudvsem.ru open API (Работа России)",
        "endpoint": API,
        "collected_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filter_created_since": args.since,
        "creation_date_span": date_span,
        "regions": list(REGIONS.values()),
        "dropped_sparse_columns": dropped_sparse,
        "n_rows": len(df), "n_cols": df.shape[1],
        "target": "Зарплата выше медианы",
        "target_threshold_rub": threshold,
        "class_balance": df["Зарплата выше медианы"].mean().round(4),
        "dropped_leaking_fields": sorted(LEAKING),
        "leakage_guard": "passed: no retained cell contains the row's salary figure",
    }
    with open(args.out.replace(".csv", "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
