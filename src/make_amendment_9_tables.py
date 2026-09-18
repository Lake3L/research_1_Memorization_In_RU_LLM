"""Print the two machine-generated tables of AMENDMENT_9 from the committed
JSON files, so that no hash, count or rate in the frozen document is
transcribed by hand.

  --files      §5: one row per ru_exposure dataset — source, published date,
               sha256, rows × columns, encoding and separator, header window,
               R2 null and minimum detectable rate at 250 queries, near-
               duplicate share at tau 0.10 (from data/registry.json,
               data/ru_exposure_profile.json, data/detectability.json)
  --exposure   §4: every registered dataset — GitHub file-name count, the two
               fragment counts with their mode, Wikipedia 2025 views
               (from data/exposure_counts.json)
"""

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ORDER = ["mkb10_v2", "okved2", "oksm", "okpdtr", "mos_metro_stations_2022",
         "mos_streets_omk_um_2022", "cardio_train", "alice_train_sessions", "telecom_churn"]


def load(name):
    return json.load(open(os.path.join(DATA, name), encoding="utf-8"))


def files_table():
    reg, det = load("registry.json"), {r["dataset"]: r for r in load("detectability.json")}
    prof = {os.path.basename(p["file"])[:-4]: p for p in load("ru_exposure_profile.json")}
    print("| dataset | source, version | published | sha256 | rows × cols | bytes | header + row 1 | null (R2) | MDR at 250 | near-dup τ=0.10 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in ORDER:
        r, d, p = reg[name], det[name], prof[name]
        src = r["source_url"].replace("https://", "")
        enc = f"{'UTF-8 with BOM' if p['bom'] else 'UTF-8'}, `{p['separator']}`"
        hw = f"{p['header_plus_first_row_chars']} {'✓' if p['header_test_applicable'] else '✗'}"
        print(f"| {name} | {src} | {r['published']} | `{r['raw_sha256'][:16]}…` | "
              f"{r['n_rows']:,} × {r['n_cols']} | {enc} | {hw} | {d['null']:.1e} | "
              f"{d['minimum_detectable_rate']:.1%} | {p['near_duplicate']['shares']['0.1']:.1%} |")


def exposure_table():
    counts = load("exposure_counts.json")
    reg = load("registry.json")
    groups = ["canon", "ru_pre_cutoff", "fresh_control", "ru_exposure"]
    print("| dataset | group | mode | GitHub: file name | GitHub: fragment 1 | GitHub: fragment 2 | ru-Wikipedia 2025 |")
    print("|---|---|---|---|---|---|---|")
    names = [n for g in groups for n in reg if reg[n]["group"] == g and n in counts]
    for name in ORDER:
        if name in names:
            names.remove(name)
    for name in list(n for n in names if reg[n]["group"] != "ru_exposure") + ORDER:
        c = counts.get(name)
        if not c:
            continue
        gh = c["github"]
        fn = [v for k, v in gh.items() if k.startswith("name:")]
        fn_s = " / ".join(f"{v['files']:,}" if isinstance(v["files"], int) else "—" for v in fn) or "—"
        fr = [gh.get("fragment_1", {}), gh.get("fragment_2", {})]
        fr_s = [f"{v.get('files'):,}" if isinstance(v.get("files"), int) else "—" for v in fr]
        w = c.get("wikipedia") or {}
        w_s = f"{w['views_2025']:,}" if isinstance(w.get("views_2025"), int) else "—"
        print(f"| {name} | {c['group']} | {c['mode']} | {fn_s} | {fr_s[0]} | {fr_s[1]} | {w_s} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", action="store_true")
    ap.add_argument("--exposure", action="store_true")
    args = ap.parse_args()
    if args.files:
        files_table()
    if args.exposure:
        exposure_table()


if __name__ == "__main__":
    main()
