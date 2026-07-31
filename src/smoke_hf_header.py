"""Pipeline smoke test: header test on Iris with a small HF model on CPU.

Validates the HF backend end-to-end (chat templating, greedy decoding,
tabmemcheck prompt flow). NOT a confirmatory run: a 0.5B model is not
expected to reproduce Table 5 of Bordt et al.; the positive-control gate
(PREREGISTRATION.md §8) runs later on stronger models.
"""

import sys
import numpy as np
import tabmemcheck as tabmem
from tabmemcheck import utils

from hf_llm import hf_setup

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "iris.csv"  # resolved from tabmemcheck resources

if __name__ == "__main__":
    tabmem.config.max_tokens = 300
    tabmem.config.print_prompts = False
    tabmem.config.print_responses = True

    print(f"loading {MODEL} on cpu ...")
    llm = hf_setup(MODEL, device="cpu")

    csv_file = utils.get_dataset_path(DATASET) if hasattr(utils, "get_dataset_path") else DATASET
    rng = np.random.default_rng(42)
    header_prompt, header_completion, response = tabmem.header_test(
        csv_file, llm, rng=rng
    )
    print("\n--- header test finished ---")
    match_len = 0
    for a, b in zip(header_completion, response):
        if a != b:
            break
        match_len += 1
    print(f"verbatim match length: {match_len} chars of {len(header_completion)}")
    sys.exit(0)
