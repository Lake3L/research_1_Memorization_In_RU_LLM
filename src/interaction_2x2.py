"""The base-vs-adapted difference, measured under both prompting modes.

Four runs make a 2x2: a base model and its Russian adaptation, each asked through
an instruction wrapper (chat) and by direct continuation (completion). The
question H1b asks — did adaptation retain, attenuate or amplify inherited
memorization — has an answer in each mode, and this reports whether those answers
agree.

Within a mode the two models receive byte-identical prompts in identical order,
because the seed and the code path are the same, so the comparison is paired and
McNemar applies. Across modes the sampled rows differ, because chat mode also
draws few-shot examples from the random stream, so the two paired differences are
estimated on different draws and the interaction is tested as the difference
between them.

Usage:
  python src/interaction_2x2.py --dataset iris.csv \\
      --base-chat  results/calls_..._raw_en_<stamp>.jsonl \\
      --base-completion results/calls_..._completion_<stamp>.jsonl \\
      --adapted-chat results/calls_..._raw_en_<stamp>.jsonl \\
      --adapted-completion results/calls_..._completion_<stamp>.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_registry import load_registry  # noqa: E402
from rescore_calls import block_index, prompt_text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hits_by_prompt(log_path, index, dataset):
    """prompt -> did the response contain the true next row, byte-exact."""
    out = {}
    for line in open(log_path, encoding="utf-8"):
        call = json.loads(line)
        if call.get("test") != "row" or call.get("dataset") != dataset:
            continue
        prompt = prompt_text(call).strip()
        truth = index.get(prompt)
        if truth is None:
            continue
        out[prompt] = truth.strip() in str(call["response"]).strip()
    return out


def paired(base, adapted):
    """McNemar over the prompts both models were asked."""
    shared = set(base) & set(adapted)
    both = sum(1 for p in shared if base[p] and adapted[p])
    base_only = sum(1 for p in shared if base[p] and not adapted[p])
    adapted_only = sum(1 for p in shared if adapted[p] and not base[p])
    n = len(shared)
    discordant = base_only + adapted_only
    p = stats.binomtest(base_only, discordant, 0.5).pvalue if discordant else 1.0
    difference = (adapted_only - base_only) / n if n else 0.0
    # standard error of a paired difference in proportions, from the discordants
    se = np.sqrt(discordant / n ** 2) if n else 0.0
    return {"n": n, "both": both, "base_only": base_only,
            "adapted_only": adapted_only, "difference": difference,
            "se": se, "p": p}


def main():
    ap = argparse.ArgumentParser()
    for arm in ("base-chat", "base-completion", "adapted-chat", "adapted-completion"):
        ap.add_argument(f"--{arm}", required=True)
    ap.add_argument("--dataset", default="iris.csv")
    ap.add_argument("--variant", default="raw")
    args = vars(ap.parse_args())

    registry = load_registry()
    name = args["dataset"][:-4]
    csv_path = os.path.join(ROOT, registry[name]["variants"][args["variant"]]["path"])
    from tabmemcheck import utils
    index = block_index(utils.load_csv_rows(csv_path))

    data = {(model, mode): hits_by_prompt(args[f"{model}_{mode}"], index, args["dataset"])
            for model in ("base", "adapted") for mode in ("chat", "completion")}

    print(f"{args['dataset']} row completion, byte-exact matches\n")
    print(f"  {'':10s} {'chat':>18s} {'completion':>18s}")
    for model in ("base", "adapted"):
        cells = []
        for mode in ("chat", "completion"):
            d = data[(model, mode)]
            cells.append(f"{sum(d.values())}/{len(d)} = {sum(d.values())/len(d):.3f}")
        print(f"  {model:10s} {cells[0]:>18s} {cells[1]:>18s}")

    print("\nPAIRED base vs adapted within each mode (McNemar; positive favours the adaptation)")
    results = {}
    for mode in ("chat", "completion"):
        r = paired(data[("base", mode)], data[("adapted", mode)])
        results[mode] = r
        ci = 1.96 * r["se"]
        print(f"  {mode:11s} n={r['n']:3d}  both={r['both']:3d}  base only={r['base_only']:3d}  "
              f"adapted only={r['adapted_only']:3d}")
        print(f"  {'':11s} adapted - base = {r['difference']:+.3f}  "
              f"95% CI [{r['difference']-ci:+.3f}, {r['difference']+ci:+.3f}]  p = {r['p']:.4f}")

    print("\nMODE effect within each model (unpaired: the modes sample different rows)")
    for model in ("base", "adapted"):
        c, k = data[(model, "chat")], data[(model, "completion")]
        hc, nc, hk, nk = sum(c.values()), len(c), sum(k.values()), len(k)
        odds, p = stats.fisher_exact([[hk, nk - hk], [hc, nc - hc]])
        print(f"  {model:10s} chat {hc}/{nc} = {hc/nc:.3f} -> completion {hk}/{nk} = {hk/nk:.3f}  "
              f"Fisher p = {p:.2e}  odds ratio {odds:.2f}")

    d_chat, d_comp = results["chat"]["difference"], results["completion"]["difference"]
    se = np.sqrt(results["chat"]["se"] ** 2 + results["completion"]["se"] ** 2)
    interaction = d_comp - d_chat
    z = interaction / se if se else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    print("\nINTERACTION — does the base-vs-adapted difference depend on how the model is asked?")
    print(f"  (adapted-base | completion) - (adapted-base | chat) = "
          f"{d_comp:+.3f} - ({d_chat:+.3f}) = {interaction:+.3f}")
    print(f"  95% CI [{interaction - 1.96*se:+.3f}, {interaction + 1.96*se:+.3f}]   "
          f"z = {z:.2f}   p = {p:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
