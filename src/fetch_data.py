"""Rebuild the frozen dataset files on a machine that has none of them.

The registry carries provenance and hashes; the CSVs themselves are gitignored
(they are other people's data, and several licences do not allow us to
redistribute them). A clean clone — Kaggle, Colab, a reviewer's laptop —
therefore has to fetch them again, and a run is only valid if the bytes it
fetched are the bytes that were frozen.

`source_url` in the registry is the *provenance* of a dataset: the page a human
should cite and visit. It is frequently not a file: UCI, OpenML and Kaggle all
serve landing pages there. This module holds the second, separate thing — where
a machine can get the exact bytes — and keeps the two apart on purpose, because
a citable source and a downloadable one are different claims.

Five of the six canon files are byte-identical to the copies shipped inside
`tabmemcheck` itself (verified), so the primary source for them is the installed
package: no download, and it is the same artefact the reference implementation
tests against. The pinned GitHub commit is the fallback.

Usage:
  python src/fetch_data.py --group canon
  python src/fetch_data.py --all --report data/fetch_report.json
"""

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import (ROOT, load_registry, read_table, sha256,  # noqa: E402
                              write_variants)

# Pinned commit of interpretml/LLM-Tabular-Memorization-Checker (v0.1.6, the
# version under test). A branch name here would silently change the bytes.
TABMEM_COMMIT = "7dbeaac57884c57806b026b955cc68f3da754171"
TABMEM_RAW = ("https://raw.githubusercontent.com/interpretml/"
              f"LLM-Tabular-Memorization-Checker/{TABMEM_COMMIT}/tabmemcheck/resources/csv")

USER_AGENT = "research-memorization-ru/1.0 (+academic use)"

# name -> ordered list of machine-readable sources.
#   ("package", filename)  the copy inside the installed tabmemcheck
#   ("url", url)           a direct download of the exact file
#   ("kaggle", slug, member) anonymous Kaggle dataset download, one member of the zip
#   ("local", reason)      cannot be re-fetched; must be carried to the machine
SOURCES = {
    "iris": [("package", "iris.csv"), ("url", f"{TABMEM_RAW}/iris.csv")],
    "uci-wine": [("package", "uci-wine.csv"), ("url", f"{TABMEM_RAW}/uci-wine.csv")],
    "openml-diabetes": [("package", "openml-diabetes.csv"),
                        ("url", f"{TABMEM_RAW}/openml-diabetes.csv")],
    "adult-train": [("package", "adult-train.csv"), ("url", f"{TABMEM_RAW}/adult-train.csv")],
    "california-housing": [("package", "california-housing.csv"),
                           ("url", f"{TABMEM_RAW}/california-housing.csv")],
    # Kaggle's own competition file needs an account; this public mirror carries
    # the same 891 rows and column order, pinned to the commit whose bytes we
    # froze. The caveat about byte-identity with Kaggle's original stands and is
    # recorded in AMENDMENT_1_DATASETS.md.
    "titanic-train": [("url", "https://raw.githubusercontent.com/datasciencedojo/datasets/"
                              "4cd38e7a532643145d00c1e512b9d16899ae9543/titanic.csv")],

    "hflabs_city": [("url", None)],          # registry URL is already a pinned raw file
    "govdomains": [("url", None)],
    "mos_zemelnye_uchastki": [("url", None)],
    "mos_torgovye_obekty": [("url", None)],
    "russian_retail": [("kaggle", "pavelkunitsyn/russian-retail", None)],

    # Collected by us from a live API; the API returns today's vacancies, so the
    # frozen file cannot be reproduced by re-querying it. It travels with the run.
    "trudvsem_vacancies_2026": [("local", "collected from a live API on 2026-08-10; "
                                          "upload the frozen CSV with the run")],
}


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _from_package(filename: str) -> bytes:
    import tabmemcheck
    path = os.path.join(os.path.dirname(tabmemcheck.__file__), "resources", "csv", filename)
    with open(path, "rb") as f:
        return f.read()


def _from_kaggle(slug: str, member) -> bytes:
    """Anonymous dataset download. Works for datasets, never for competitions."""
    data = _download(f"https://www.kaggle.com/api/v1/datasets/download/{slug}")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        name = member or (names[0] if len(names) == 1 else None)
        if name is None:
            raise ValueError(f"{slug}: ambiguous archive, members={names}")
        return archive.read(name)


def obtain(name: str, rec: dict) -> tuple:
    """Return (bytes, source_label) for one dataset, or raise."""
    errors = []
    for spec in SOURCES.get(name, [("url", None)]):
        kind = spec[0]
        try:
            if kind == "package":
                return _from_package(spec[1]), f"tabmemcheck package: {spec[1]}"
            if kind == "url":
                url = spec[1] or rec["source_url"]
                return _download(url), f"url: {url}"
            if kind == "kaggle":
                return _from_kaggle(spec[1], spec[2]), f"kaggle: {spec[1]}"
            if kind == "local":
                raise RuntimeError(f"not re-fetchable — {spec[1]}")
        except Exception as e:  # try the next source; report all failures if none works
            errors.append(f"{kind}: {type(e).__name__}: {e}")
    raise RuntimeError("; ".join(errors) or "no source configured")


def fetch_one(name: str, rec: dict, rebuild_variants: bool = True) -> dict:
    """Materialise one dataset and check it against the frozen hashes."""
    path = os.path.join(ROOT, rec["raw_path"])
    result = {"dataset": name, "path": rec["raw_path"]}

    if os.path.exists(path) and sha256(path) == rec["raw_sha256"]:
        result.update(status="cached", source="already on disk")
    else:
        try:
            payload, source = obtain(name, rec)
        except Exception as e:
            result.update(status="unavailable", error=str(e))
            return result
        digest = __import__("hashlib").sha256(payload).hexdigest()
        if digest != rec["raw_sha256"]:
            # Do not overwrite the frozen path with bytes that are not the frozen
            # bytes: a later run must not silently inherit the wrong file.
            quarantine = path + ".mismatch"
            os.makedirs(os.path.dirname(quarantine), exist_ok=True)
            with open(quarantine, "wb") as f:
                f.write(payload)
            result.update(status="hash_mismatch", source=source, got=digest[:16],
                          expected=rec["raw_sha256"][:16], quarantined=quarantine)
            return result
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(payload)
        result.update(status="fetched", source=source)

    if rebuild_variants:
        result["variants"] = rebuild(name, rec, path)
    return result


def rebuild(name: str, rec: dict, raw_path: str) -> dict:
    """Regenerate the derived serialisations and check them against the freeze.

    A mismatch here is not a corrupted download — it means this machine's pandas
    serialises differently from the machine that froze the registry. That would
    make per-variant results incomparable across runs, so it has to surface.
    """
    out_dir = os.path.join(ROOT, "data", rec["group"], "variants")
    made = write_variants(read_table(raw_path), name, out_dir, raw_path=raw_path)
    report = {}
    for variant, frozen in rec.get("variants", {}).items():
        if variant not in made:
            report[variant] = "not produced on this machine"
        elif made[variant]["sha256"] != frozen["sha256"]:
            report[variant] = "differs from the frozen hash"
        else:
            report[variant] = "ok"
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None, help="canon | ru_pre_cutoff | fresh_control")
    ap.add_argument("--only", default=None, help="comma-separated dataset names")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-variants", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    registry = load_registry()
    targets = registry
    if args.group:
        targets = {k: v for k, v in targets.items() if v["group"] == args.group}
    if args.only:
        keep = set(args.only.split(","))
        targets = {k: v for k, v in targets.items() if k in keep}
    if not targets:
        sys.exit("no dataset selected")

    results = [fetch_one(n, r, rebuild_variants=not args.no_variants)
               for n, r in targets.items()]

    for r in results:
        bad = [f"{k}={v}" for k, v in (r.get("variants") or {}).items() if v != "ok"]
        print(f"{r['dataset']:26s} {r['status']:14s} {r.get('source', r.get('error',''))[:60]}"
              + (f"  [variants: {', '.join(bad)}]" if bad else ""))

    ok = [r for r in results if r["status"] in ("cached", "fetched")]
    print(f"\n{len(ok)}/{len(results)} datasets available with the frozen bytes")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("wrote", args.report)

    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
