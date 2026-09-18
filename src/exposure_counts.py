"""The exposure covariate of AMENDMENT_9 (expedition E1 of AMENDMENT_8).

"Exposure" is how often a table occurs in public text a model could have been
trained on. Kandpal et al. (arXiv:2202.06539) show that a sequence seen once
is almost never regurgitated and one seen ten times is regurgitated orders of
magnitude more often, so a memorization verdict is only interpretable next to
a count. Two counts are taken here for every frozen dataset, by one rule,
before any model runs on the new files:

  github    the number of files GitHub code search returns for an exact
            string, through the authenticated `gh` CLI (10 requests/min):
            (a) the file's canonical name, when it has one;
            (b) two fragments taken by rule from the file — the records at
                positions 10 and floor(N/2) (1-based, header excluded), the
                whole physical line, capped at its first 200 characters, for
                a table that circulates as a CSV file (`full_line`), or the
                content-witness field alone for a
                classifier whose content is reprinted without its CSV shape
                (`name_column`).
  wikipedia the 2025 calendar-year user pageviews of the Russian Wikipedia
            article whose subject the table is, from the Wikimedia REST API;
            titles are fixed here, canonical (not redirects).

GitHub's index excludes forks, archived repositories and files over 384 KB,
so every count is a floor. Output: data/exposure_counts.json.

Usage:
  python src/exposure_counts.py                # every registered dataset
  python src/exposure_counts.py --only okved2,cardio_train
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry  # noqa: E402
from metrics import infer_separator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "exposure_counts.json")
POSITIONS = (10, "half")
FRAGMENT_CAP = 200   # characters; GitHub rejects queries over 256 characters
PAUSE = 7  # seconds between GitHub code-search calls

# how each table circulates, and its canonical file name(s) if it has one
MODE = {
    "name_column": ["mkb10_v2", "okved2", "oksm", "okpdtr", "mos_metro_stations_2022",
                    "mos_streets_omk_um_2022"],
}
FILE_NAMES = {
    "iris": ["iris.csv"], "uci-wine": ["wine.csv"], "openml-diabetes": ["diabetes.csv"],
    "adult-train": ["adult.csv"], "california-housing": ["housing.csv"],
    "titanic-train": ["titanic.csv"],
    "hflabs_city": ["city.csv"], "govdomains": ["feddomains.csv"],
    "cardio_train": ["cardio_train.csv", "mlbootcamp5_train.csv"],
    "alice_train_sessions": ["train_sessions.csv"],
    "telecom_churn": ["telecom_churn.csv"],
}
NAME_COLUMN = {
    "mkb10_v2": "Наименование", "okved2": "Наименование",
    "oksm": "Полное наименование по ОКСМ", "okpdtr": "Наименование",
    "mos_metro_stations_2022": "Наименование станции",   # second header line names
    "mos_streets_omk_um_2022": "Полное наименование",
}
WIKIPEDIA = {
    "iris": "Ирисы Фишера", "titanic-train": "Титаник",
    "mkb10_v2": "МКБ-10",
    "okved2": "Общероссийский классификатор видов экономической деятельности",
    "oksm": "Общероссийский классификатор стран мира",
    "okpdtr": "Общероссийский классификатор профессий рабочих, должностей служащих и тарифных разрядов",
    "mos_metro_stations_2022": "Список станций Московского метрополитена",
    "mos_streets_omk_um_2022": "Список улиц Москвы",
    "hflabs_city": "Список городов России",
}


def gh_count(query: str):
    r = subprocess.run(["gh", "api", "-X", "GET", "search/code", "-f", f"q={query}",
                        "-f", "per_page=1"], capture_output=True, text=True, encoding="utf-8")
    try:
        d = json.loads(r.stdout)
        return d["total_count"], bool(d.get("incomplete_results"))
    except (json.JSONDecodeError, KeyError):
        return None, r.stderr.strip()[:200]


def wiki_views(title: str, year: int = 2025):
    u = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/ru.wikipedia/"
         "all-access/user/" + urllib.parse.quote(title.replace(" ", "_"), safe="")
         + f"/monthly/{year}010100/{year}123100")
    req = urllib.request.Request(u, headers={"User-Agent": "research-memorization-ru/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as f:
            items = json.load(f)["items"]
        return sum(x["views"] for x in items), len(items)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}"


def fragments(path: str, mode: str, name_column):
    text = open(path, encoding="utf-8-sig").read()
    lines = [l for l in text.splitlines() if l.strip()]
    header, body = lines[0], lines[1:]
    if body and not any(ch.isdigit() for ch in body[0].replace('"', "")[:40]) and \
            len(body) > 1 and "global_id" in header:
        body = body[1:]                      # data.mos.ru: skip the Russian header line
    n = len(body)
    picks = [min(POSITIONS[0], n) - 1, n // 2 - 1]
    out = []
    sep = infer_separator(header)
    if mode == "full_line":
        for i in picks:
            out.append(body[i][:FRAGMENT_CAP])   # a prefix of the row is still the row's text
    else:
        # name_column: the field of the content witness; column found by header name,
        # or by the Russian header line of a Moscow export
        head_fields = next(csv.reader([header], delimiter=sep))
        if name_column in head_fields:
            j = head_fields.index(name_column)
        else:
            ru = next(csv.reader([lines[1]], delimiter=sep))
            j = ru.index(name_column)
        for i in picks:
            out.append(next(csv.reader([body[i]], delimiter=sep))[j].strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    registry = load_registry()
    names = args.only.split(",") if args.only else list(registry)
    existing = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    existing.pop("_what", None)

    for name in names:
        rec = registry[name]
        path = os.path.join(ROOT, rec["variants"]["raw"]["path"])
        mode = "name_column" if name in MODE["name_column"] else "full_line"
        entry = {"group": rec["group"], "mode": mode, "measured": date.today().isoformat(),
                 "github": {}, "wikipedia": None}
        for fname in FILE_NAMES.get(name, []):
            q = f'"{fname}"'
            n, flag = gh_count(q)
            entry["github"][f"name:{fname}"] = {"query": q, "files": n, "incomplete": flag}
            print(f"{name:26} {n!s:>8}  {q}", flush=True)
            time.sleep(PAUSE)
        for k, frag in enumerate(fragments(path, mode, NAME_COLUMN.get(name))):
            q = '"' + frag.replace('"', '\\"') + '"'
            n, flag = gh_count(q)
            entry["github"][f"fragment_{k + 1}"] = {"query": q, "files": n, "incomplete": flag}
            print(f"{name:26} {n!s:>8}  {q[:90]}", flush=True)
            time.sleep(PAUSE)
        if name in WIKIPEDIA:
            views, months = wiki_views(WIKIPEDIA[name])
            entry["wikipedia"] = {"title": WIKIPEDIA[name], "views_2025": views, "months": months}
            print(f"{name:26} {views!s:>8}  wikipedia: {WIKIPEDIA[name]}", flush=True)
        existing[name] = entry
        out = {"_what": __doc__.split("\n\n")[1].strip(), **existing}
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
