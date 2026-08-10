"""Registry of datasets under test: provenance, hashes, and serialisation variants.

Two jobs.

*Provenance.* Every dataset carries where it came from, the evidence for its
publication date, when we downloaded it, and a SHA-256 of the exact bytes. A
memorization claim is only as good as the claim that the file predates the
model, so that evidence is part of the data, not the prose.

*Serialisation variants.* This is the part with no counterpart in Bordt et al.
Their canon has one unambiguous CSV form each: UTF-8, comma-separated, a known
column order. Russian open data does not. The same table circulates as cp1251
and UTF-8, comma- and semicolon-separated, with `,` or `.` as the decimal mark.
Verbatim memorization is a property of the exact byte string the model saw, so
testing a single guessed form risks reporting "no memorization" when we simply
guessed the wrong one. We therefore materialise the plausible forms, hash each,
and let the tests run over all of them; the paper reports per-variant results
and treats the maximum as the extraction estimate.

The `raw` variant is the published file itself and is never regenerated. It has
to be listed explicitly, because the derived variants are pandas round-trips and
a round-trip is *not* byte-preserving: iris ships `4.9,3,1.4,0.2` and pandas
writes `4.9,3.0,1.4,0.2`; on uci-wine 99.4% of rows differ from the published
bytes this way. Running a verbatim test only against derived variants would
therefore score a perfectly memorized canon dataset near zero. `raw` is the form
Bordt et al. tested and the only one comparable to their published counts.

Paths are stored POSIX-style relative to the project root: the registry is
written on Windows and read on Linux (Kaggle/Colab), and a backslash path is not
a path there.
"""

import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
REGISTRY_PATH = os.path.join(DATA, "registry.json")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: str) -> str:
    """Path relative to the project root, POSIX-style, so Linux can read it."""
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


@dataclass
class DatasetRecord:
    name: str
    group: str                    # canon | ru_pre_cutoff | fresh_control
    source_url: str
    published: Optional[str]      # ISO date of public availability
    published_evidence: str       # how we know: archive link, API field, commit
    license: str
    downloaded_utc: str
    raw_path: str
    raw_sha256: str
    n_rows: int
    n_cols: int
    columns: list
    language: str                 # ru | en | mixed — of headers and values
    variants: dict = field(default_factory=dict)     # variant name -> {path, sha256}
    diagnostics: dict = field(default_factory=dict)  # entropy / duplicate stats
    notes: str = ""


def read_table(path: str) -> pd.DataFrame:
    """Read a CSV whose encoding and separator we do not know in advance."""
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            with open(path, encoding=encoding) as f:
                sample = f.read(8192)
            sep = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
            return pd.read_csv(path, encoding=encoding, sep=sep)
        except (UnicodeDecodeError, csv.Error):
            continue
    raise ValueError(f"cannot read {path}: unknown encoding or separator")


def diagnostics(df: pd.DataFrame) -> dict:
    """Statistics the memorization tests need in order to be interpretable.

    Duplicate rate is the base rate a row-completion count must beat. Digits per
    row is Ward et al.'s covariate — longer digit strings are harder to
    reproduce, so it must be comparable across datasets before their counts are.
    """
    rows = df.astype(str).apply(lambda r: ",".join(r), axis=1)
    unique_share = {c: df[c].nunique() / max(len(df), 1) for c in df.columns}
    most_unique = max(unique_share, key=unique_share.get) if len(df.columns) else None
    return {
        "duplicate_row_share": round(1 - rows.nunique() / max(len(rows), 1), 4),
        "mean_digits_per_row": round(
            float(rows.str.count(r"\d").mean()) if len(rows) else 0.0, 1),
        "mean_chars_per_row": round(float(rows.str.len().mean()) if len(rows) else 0.0, 1),
        "most_unique_feature": most_unique,
        "most_unique_share": round(unique_share.get(most_unique, 0), 4) if most_unique else None,
        "high_entropy_feature_available": bool(
            most_unique and unique_share[most_unique] > 0.5),
        "cyrillic_in_headers": any(_has_cyrillic(str(c)) for c in df.columns),
        "cyrillic_in_values": bool(
            df.select_dtypes(include=["object"]).map(
                lambda v: _has_cyrillic(str(v))).any().any()
        ) if not df.select_dtypes(include=["object"]).empty else False,
    }


def _has_cyrillic(s: str) -> bool:
    return any("Ѐ" <= ch <= "ӿ" for ch in s)


def write_variants(df: pd.DataFrame, name: str, out_dir: str,
                   raw_path: Optional[str] = None) -> dict:
    """Materialise the serialisations a Russian table plausibly circulates in.

    `raw_path`, when given, is listed first and untouched: it is the published
    file, and the derived variants below are pandas round-trips of it, which is
    a different byte string (see the module docstring).
    """
    os.makedirs(out_dir, exist_ok=True)
    variants = {}
    if raw_path is not None:
        variants["raw"] = {"path": rel(raw_path), "sha256": sha256(raw_path)}
    specs = {
        "utf8_comma": dict(encoding="utf-8", sep=",", decimal="."),
        "utf8_semicolon": dict(encoding="utf-8", sep=";", decimal="."),
        "cp1251_semicolon": dict(encoding="cp1251", sep=";", decimal=","),
        "utf8_semicolon_decimal_comma": dict(encoding="utf-8", sep=";", decimal=","),
    }
    for variant, spec in specs.items():
        path = os.path.join(out_dir, f"{name}__{variant}.csv")
        buf = io.StringIO()
        df.to_csv(buf, index=False, sep=spec["sep"], decimal=spec["decimal"],
                  lineterminator="\n")
        try:
            with open(path, "w", encoding=spec["encoding"], newline="") as f:
                f.write(buf.getvalue())
        except UnicodeEncodeError:
            continue  # cp1251 cannot hold every character; skip rather than mangle
        variants[variant] = {"path": rel(path), "sha256": sha256(path)}
    return variants


def register(name, group, source_url, published, published_evidence, license,
             raw_path, language, notes="") -> DatasetRecord:
    df = read_table(raw_path)
    record = DatasetRecord(
        name=name, group=group, source_url=source_url, published=published,
        published_evidence=published_evidence, license=license,
        downloaded_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        raw_path=rel(raw_path), raw_sha256=sha256(raw_path),
        n_rows=len(df), n_cols=df.shape[1], columns=list(map(str, df.columns)),
        language=language,
        variants=write_variants(df, name, os.path.join(DATA, group, "variants"),
                                raw_path=raw_path),
        diagnostics=diagnostics(df), notes=notes,
    )
    registry = load_registry()
    registry[name] = asdict(record)
    os.makedirs(DATA, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    return record


def load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        return json.load(open(REGISTRY_PATH, encoding="utf-8"))
    return {}


def normalise() -> list:
    """Bring an already-frozen registry to the current conventions, in place.

    Idempotent and hash-preserving: it rewrites Windows paths as POSIX and adds
    the `raw` variant entry, whose hash is the already-frozen `raw_sha256`. No
    dataset is added, removed or re-downloaded, and no existing hash changes —
    otherwise this would be a new freeze rather than a migration of the old one.
    """
    registry = load_registry()
    changes = []
    for name, rec in registry.items():
        posix_raw = rec["raw_path"].replace("\\", "/")
        if posix_raw != rec["raw_path"]:
            changes.append(f"{name}: raw_path -> {posix_raw}")
            rec["raw_path"] = posix_raw
        variants = rec.get("variants", {})
        for variant, info in variants.items():
            posix = info["path"].replace("\\", "/")
            if posix != info["path"]:
                changes.append(f"{name}/{variant}: path -> {posix}")
                info["path"] = posix
        if "raw" not in variants:
            changes.append(f"{name}: + variant 'raw' (the published bytes)")
            rec["variants"] = {"raw": {"path": posix_raw,
                                       "sha256": rec["raw_sha256"]}, **variants}
    if changes:
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
    return changes


def verify() -> list:
    """Re-hash every registered file. Any mismatch invalidates results built on it."""
    problems = []
    for name, rec in load_registry().items():
        raw = os.path.join(ROOT, rec["raw_path"])
        if not os.path.exists(raw):
            problems.append(f"{name}: raw file missing ({rec['raw_path']})")
        elif sha256(raw) != rec["raw_sha256"]:
            problems.append(f"{name}: raw file changed since registration")
        for variant, info in rec.get("variants", {}).items():
            p = os.path.join(ROOT, info["path"])
            if not os.path.exists(p):
                problems.append(f"{name}/{variant}: variant missing")
            elif sha256(p) != info["sha256"]:
                problems.append(f"{name}/{variant}: variant changed since registration")
    return problems


if __name__ == "__main__":
    import sys

    if "--normalise" in sys.argv:
        for change in normalise() or ["nothing to change"]:
            print(" ", change)
    problems = verify()
    registry = load_registry()
    print(f"registered datasets: {len(registry)}")
    for name, rec in registry.items():
        d = rec["diagnostics"]
        print(f"  {name:28s} {rec['group']:15s} {rec['n_rows']:>7d} rows  "
              f"dup={d['duplicate_row_share']:.3f}  "
              f"entropy_feature={d['most_unique_feature']}")
    print("integrity:", "OK" if not problems else "\n  ".join([""] + problems))
