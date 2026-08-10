"""Generate the dataset amendment to the preregistration from the registry itself.

Written by machine from registry.json and the quality reports so the hashes and
counts in the frozen document cannot drift from the files they describe.
"""

import json
import os
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "data", "registry.json")

H3_ADMISSION = {}  # filled from *_quality.json


def load_quality():
    for group in ("ru_pre_cutoff", "fresh_control"):
        folder = os.path.join(ROOT, "data", group)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith("_quality.json"):
                q = json.load(open(os.path.join(folder, f), encoding="utf-8"))
                name = f.replace("__modeling_quality.json", "").replace("_quality.json", "")
                H3_ADMISSION[name] = q


def main():
    load_quality()
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    today = date.today().isoformat()

    lines = [
        "# Amendment 1 to PREREGISTRATION.md — dataset list frozen",
        "",
        f"**Date:** {today}. **Status:** frozen before any model was queried on any of "
        "these files.",
        "",
        "PREREGISTRATION.md §4 committed to freezing the Russian dataset list, its "
        "canonicalisation and SHA-256 hashes in a dated amendment before any model saw "
        "them. This is that amendment. It is generated from `data/registry.json` by "
        "`src/make_amendment.py`, so the hashes below are the hashes of the files on "
        "disk, not transcribed by hand.",
        "",
        "## Admission to H3",
        "",
        "§4 admits a dataset on row count, language, an available high-entropy feature "
        "and evidence of publication date. Those criteria say nothing about whether the "
        "classification task the dataset defines is learnable at all — and a dataset "
        "whose task is not learnable cannot support H3, because few-shot accuracy on it "
        "would be uninformative whether or not the model memorized it. We therefore add "
        "one criterion, decided before any model was run: **a dataset enters H3 only if "
        "logistic regression or gradient boosting beats the majority class by at least "
        "5 accuracy points** under stratified shuffled 5-fold cross-validation "
        "(`src/check_dataset.py`).",
        "",
        "Datasets that fail this remain in H1/H2. The memorization tests ask whether the "
        "model reproduces rows verbatim; they need no target and are unaffected.",
        "",
        "## Frozen datasets",
        "",
    ]

    for group, title in (("ru_pre_cutoff", "Russian, published before 2023-01-01"),
                         ("fresh_control", "Fresh control, collected after 2026-06-01"),
                         ("canon", "Western canon (reference, unchanged from Bordt et al.)")):
        members = {k: v for k, v in reg.items() if v["group"] == group}
        if not members:
            continue
        lines += [f"### {title}", ""]
        for name, rec in members.items():
            q = H3_ADMISSION.get(name, {})
            base = q.get("baselines", {})
            admitted = q.get("admitted")
            d = rec["diagnostics"]
            lines += [
                f"**`{name}`** — {rec['n_rows']} rows × {rec['n_cols']} columns",
                "",
                f"- source: {rec['source_url']}",
                f"- published: {rec['published']} — {rec['published_evidence']}",
                f"- license: {rec['license']}",
                f"- downloaded: {rec['downloaded_utc']}",
                f"- SHA-256 (published file): `{rec['raw_sha256']}`",
                f"- Cyrillic headers: {d['cyrillic_in_headers']}; Cyrillic values: "
                f"{d['cyrillic_in_values']}",
                f"- duplicate rows: {d['duplicate_row_share']:.4f}; digits per row: "
                f"{d['mean_digits_per_row']}; highest-entropy feature: "
                f"`{d['most_unique_feature']}` ({d['most_unique_share']:.0%} unique)",
            ]
            if base:
                lines.append(
                    f"- baselines: LR {base['logistic_regression']['accuracy']:.3f} "
                    f"(κ={base['logistic_regression']['cohen_kappa']:.2f}), "
                    f"GBT {base['gradient_boosting']['accuracy']:.3f}, majority "
                    f"{base['majority_baseline']:.3f} → lift "
                    f"{base['best_lift_over_majority']:+.3f}")
                lines.append(f"- **H3: {'admitted' if admitted else 'excluded — task not '
                                        'learnable; used for H1/H2 only'}**")
            if rec.get("variants"):
                lines.append("- serialisation variants and hashes:")
                for variant, info in rec["variants"].items():
                    lines.append(f"  - `{variant}`: `{info['sha256'][:32]}…`")
            if rec.get("notes"):
                lines.append(f"- notes: {rec['notes']}")
            lines.append("")

    lines += [
        "## Canonicalisation",
        "",
        "Verbatim memorization is a property of the exact bytes a model saw, and these "
        "files have no single canonical form: the same table circulates as cp1251 and "
        "UTF-8, comma- and semicolon-separated, with `.` or `,` as the decimal mark. "
        "Testing one guessed form risks reporting absence of memorization when we merely "
        "guessed wrong. Every dataset is therefore materialised in each serialisation "
        "listed above, all of them are tested, results are reported per variant, and the "
        "maximum across variants is the extraction estimate. This has no counterpart in "
        "Bordt et al., whose canon has one unambiguous form per dataset.",
        "",
        "The `raw` variant is the published file itself, listed separately because the "
        "other variants are pandas round-trips and a round-trip does not preserve bytes: "
        "iris ships `4.9,3,1.4,0.2` and pandas writes `4.9,3.0,1.4,0.2`; on `uci-wine` "
        "99.4% of rows and on `titanic-train` 79.6% differ from the published bytes for "
        "reasons of float formatting alone. A verbatim test run only against derived "
        "variants would therefore score a perfectly memorized dataset near zero. `raw` is "
        "the form Bordt et al. tested and the only one whose counts are comparable with "
        "their published tables; it is what the §8 gate and all canon results use.",
        "",
        "## Deviations from §4 as written",
        "",
        "- §4 set a target of at least four Russian pre-cutoff datasets with a minimum of "
        "two. Five are frozen here, of which three carry a learnable task and enter H3.",
        "- §4 named Kaggle, data.gov.ru, data.mos.ru and Rosstat as candidate sources. "
        "data.gov.ru is now an empty single-page application with no catalogue or API, "
        "rosstat.gov.ru did not respond, and data.mos.ru serves data only through an API "
        "requiring a key. Two datasets here come from a 2013-2014 GitHub mirror of "
        "data.mos.ru instead, which has the side benefit of a provable publication date.",
        "- One dataset (`russian_retail`) comes from Kaggle under an unclear licence "
        "('Other, specified in description'). It is used for memorization testing only, "
        "no redistribution, and this is flagged in the paper's data statement.",
    ]

    out = os.path.join(ROOT, "AMENDMENT_1_DATASETS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
