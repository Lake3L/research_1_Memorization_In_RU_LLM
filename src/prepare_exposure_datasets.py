"""Freeze the high-exposure Russian tables of AMENDMENT_9: copy the published
bytes into data/ru_exposure/, register provenance and hashes, designate the
witness columns.

The files were collected outside the repository (see LOG.md 2026-09-18) and are
copied here byte for byte: nothing is re-encoded, re-separated or re-ordered,
because the memorization tests are about the text a model could have seen.
The only test-relevant peculiarity is recorded per file in `notes` (two header
lines in the data.mos.ru exports, a BOM in the St Petersburg exports).

Witness columns come in two tiers (AMENDMENT_9 §3):
  file     a value that exists only in this file (a portal-assigned identifier);
           reproducing it means the file itself was seen;
  content  the official name or measurement that every copy of the table
           carries; reproducing it means the table's content was memorised,
           whichever copy the model saw.
Both are designated here, before any model runs, and both are reported.

Usage:
  python src/prepare_exposure_datasets.py --staging "D:\\Загрузки\\data_rus"
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import register  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "ru_exposure")
WITNESS_FILE = os.path.join(ROOT, "data", "witness_columns.json")

SPB_LICENCE = "not stated on data.gov.spb.ru (checked 2026-09-18)"
MOS_LICENCE = ("reported as CC BY 4.0 (portal footer link to creativecommons.org/licenses/by/4.0/"
               "deed.ru, delegated check of 2026-09-18); the terms page is a client-side "
               "application and could not be read by machine — verify before publication")


def spb(dataset_id, slug, version, version_date, title, staging_name, witness, notes=""):
    return {
        "staging": os.path.join("spb", staging_name),
        "source_url": f"https://data.gov.spb.ru/irsi/{slug}/versions/{version}/export_data/",
        "published": version_date,
        "published_evidence": (
            f"data.gov.spb.ru dataset {dataset_id} ({slug}), version {version} dated "
            f"{version_date} — the latest version before 2023-01-01 in "
            f"/api/v1/datasets/{dataset_id}/versions/; the CSV inside the export zip carries "
            f"a {version_date} modification time; fetched 2026-09-18"),
        "license": SPB_LICENCE, "language": "ru",
        "witness": witness,
        "notes": f"{title}. Published bytes of the portal export: UTF-8 with BOM, comma, "
                 f"one Russian header line. " + notes,
    }


SOURCES = {
    "mkb10_v2": spb(38, "7815000870-MKB_new", 1, "2019-05-23",
                    "МКБ-10, версия 2 (International Classification of Diseases, 10th "
                    "revision, Russian adaptation)",
                    "38_7815000870-MKB_new_v1_data__MKB-10.-Versiya-2.csv",
                    {"content": ["Наименование"]},
                    "Group A. Witness (content): the official Russian name of a code."),
    "okved2": spb(20, "7815000870-OKVED_new", 19, "2020-06-22",
                  "ОКВЭД 2 (ОК 029-2014), Russian classifier of economic activities",
                  "20_7815000870-OKVED_new_v19_data__OKVED2.csv",
                  {"content": ["Наименование"]},
                  "Group A. The first record carries a long 'Описание', so header + first "
                  "row is 929 characters: the header test is not applicable (AMENDMENT_6 §3)."),
    "oksm": spb(108, "7815000870-OKSM", 4, "2019-07-29",
                "ОКСМ (ОК 025-2001), Russian classifier of countries",
                "108_7815000870-OKSM_v4_data__OKSM.csv",
                {"content": ["Полное наименование по ОКСМ", "Краткое наименование по ОКСМ"]},
                "Group A. Content overlaps ISO 3166; a positive is an upper bound on what "
                "world knowledge alone reproduces."),
    "okpdtr": spb(15, "7815000870-OKPDTR", 1, "2019-05-18",
                  "ОКПДТР (ОК 016-94), Russian classifier of occupations and positions",
                  "15_7815000870-OKPDTR_v1_data__OKPDTR.csv",
                  {"content": ["Наименование"]},
                  "Group A."),
    "mos_metro_stations_2022": {
        "staging": os.path.join("mos_pre2023", "1488_v3_rel14", "data-1488-19-12-2022.csv"),
        "source_url": "https://data.mos.ru/opendata/1488",
        "published": "2022-12-19",
        "published_evidence": (
            "data.mos.ru dataset 1488 (7704786030-MoscowSubwayStations, first published "
            "2015-12-01): version 3, release 14 dated 2022-12-19 — the latest release before "
            "2023-01-01 in GET /api/v2/odata/datasets/1488/releases; the CSV was regenerated "
            "from that release by POST /api/v2/odataExports/export on 2026-09-18"),
        "license": MOS_LICENCE, "language": "ru",
        "witness": {"file": ["global_id"], "content": ["Наименование линии", "Район"]},
        "notes": ("Станции Московского метрополитена, release of 2022-12-19: 305 stations. "
                  "Published bytes of the portal export: UTF-8 with BOM, semicolon, two "
                  "header lines (English field names, then Russian); the library reads the "
                  "Russian header line as the first row. Group C."),
    },
    "mos_streets_omk_um_2022": {
        "staging": os.path.join("mos_pre2023", "2044_v1_rel62", "data-2044-18-11-2022.csv"),
        "source_url": "https://data.mos.ru/opendata/2044",
        "published": "2022-11-18",
        "published_evidence": (
            "data.mos.ru dataset 2044 (7710168515-omk0012013razdel1, first published "
            "2015-11-30): version 1, release 62 dated 2022-11-18 — the latest release before "
            "2023-01-01 in GET /api/v2/odata/datasets/2044/releases; the CSV was regenerated "
            "from that release by POST /api/v2/odataExports/export on 2026-09-18"),
        "license": MOS_LICENCE, "language": "ru",
        "witness": {"file": ["global_id"], "content": ["UM_CODE"]},
        "notes": ("ОМК УМ (ОМК 001-2013), section 1, release of 2022-11-18: 5,405 Moscow "
                  "streets. Same export shape as the metro file (two header lines). "
                  "Alphabetical by name: AMENDMENT_7 R1 and R5 apply. Group C."),
    },
    "cardio_train": {
        "staging": os.path.join("Cardiovascular Disease dataset", "cardio_train.csv"),
        "source_url": "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset",
        "published": "2019-01-20",
        "published_evidence": (
            "Kaggle dataset created 2019-01-20 (creationDate in the Wayback capture of "
            "2019-07-15, which also lists contentLength 2941524, rowCount 70000 and "
            "contentMD5 ccRqcPe6nj5HNq5z7NW6yQ== — the MD5 of this file); byte-identical to "
            "data/mlbootcamp5_train.csv in github.com/Yorko/mlcourse.ai at commit "
            "0b6125c37cf6693ae0a698913456c3cf56058e54 (2021-12-09)"),
        "license": "Unknown (Kaggle licence field); origin Mail.Ru ML Boot Camp V (2017), "
                   "not attributed on the Kaggle page",
        "language": "en",
        "witness": {"content": ["age", "weight"]},
        "notes": ("Cardiovascular Disease dataset (Svetlana Ulianova), 70,000 rows, "
                  "semicolon. Group B: English header, Russian-origin course data."),
    },
    "alice_train_sessions": {
        "staging": os.path.join("Catch_me_If_you_can", "train_sessions.csv"),
        "source_url": "https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-"
                      "through-webpage-session-tracking2",
        "published": "2017-09-11",
        "published_evidence": (
            "Kaggle in-class competition 7173 launched 2017-09-11 (dateEnabled in the Wayback "
            "capture of 2022-12-08, which names train_sessions.csv among the files); the "
            "companion site_dic.pkl is byte-identical to the copy in github.com/Yorko/"
            "mlcourse.ai; the CSV itself is served only by Kaggle and travels with the run"),
        "license": "competition rules (kaggle.com/terms); no open-data licence",
        "language": "en",
        "witness": {"content": ["time1", "time2", "time3"]},
        "notes": ("Catch Me If You Can ('Alice'), mlcourse.ai, 253,561 sessions. "
                  "Group B."),
    },
    "telecom_churn": {
        "staging": os.path.join("mlcourse", "telecom_churn.csv"),
        "source_url": "https://raw.githubusercontent.com/Yorko/mlcourse.ai/"
                      "0b6125c37cf6693ae0a698913456c3cf56058e54/data/telecom_churn.csv",
        "published": "2021-12-09",
        "published_evidence": (
            "last commit touching the file in github.com/Yorko/mlcourse.ai: "
            "0b6125c37cf6693ae0a698913456c3cf56058e54 dated 2021-12-09 (GitHub API); the file "
            "is the topic-1 table of the course, whose repository dates from 2017-02-27"),
        "license": "not stated for the data file; course repository under its own licence",
        "language": "en",
        "witness": {"content": ["Total day minutes", "Total eve minutes",
                                "Total night minutes", "Total intl minutes"]},
        "notes": ("Telecom churn (the classic churn table as shipped by mlcourse.ai), 3,333 "
                  "rows. Group B; the charge columns are derived from the minutes and are "
                  "not witnesses."),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True, help="folder holding the collected files")
    ap.add_argument("--only", default=None, help="comma-separated dataset names")
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    only = set(args.only.split(",")) if args.only else set(SOURCES)

    witness = json.load(open(WITNESS_FILE, encoding="utf-8"))
    tiers = witness.setdefault("_tiers", {})
    for name, spec in SOURCES.items():
        if name not in only:
            continue
        src = os.path.join(args.staging, spec["staging"])
        raw_path = os.path.join(RAW, f"{name}.csv")
        shutil.copyfile(src, raw_path)          # bytes, not a decode/encode round trip
        record = register(
            name=name, group="ru_exposure", source_url=spec["source_url"],
            published=spec["published"], published_evidence=spec["published_evidence"],
            license=spec["license"], raw_path=raw_path, language=spec["language"],
            notes=spec["notes"],
        )
        cols = spec["witness"]
        witness[name] = cols.get("file", []) + cols.get("content", [])
        tiers[name] = cols
        print(f"{name}: {record.n_rows}x{record.n_cols}, sha256 {record.raw_sha256[:16]}…, "
              f"witness {witness[name]}")

    with open(WITNESS_FILE, "w", encoding="utf-8") as f:
        json.dump(witness, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
