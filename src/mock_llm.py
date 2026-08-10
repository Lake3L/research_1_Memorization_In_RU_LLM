"""Mock models that validate the measurement code before any money is spent.

Two controls for our own pipeline (not for any real model):

- PerfectMemorizer answers every query with the ground truth taken from the
  CSV. Every test must report ~100% matches. If it does not, our counting or
  parsing is wrong, and any real-model number would be meaningless.
- FormatEchoMock answers with the format of the last few-shot example, i.e.
  well-formed but wrong content. Every test must report ~0% matches (up to
  duplicate rows). This catches the opposite failure: counting code that
  credits matches it should not.

Both also give realistic response lengths, so a dry run priced with them
reflects real token usage instead of the max_tokens worst case.
"""

from typing import List, Dict

import tabmemcheck
from tabmemcheck import utils


class MockAdapter(tabmemcheck.LLM_Interface):
    """Presents a mock function to tabmemcheck as if it were a chat model.

    The mocks are plain callables so that they can be reasoned about; the tests
    want an LLM_Interface. This is the whole of the difference.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.chat_mode = True
        self.n_calls = 0

    def chat_completion(self, messages, temperature, max_tokens):
        self.n_calls += 1
        return self.fn(messages)

    def completion(self, prompt, temperature, max_tokens):
        raise NotImplementedError("mocks are chat-mode only")


def format_echo_mock(messages: List[Dict]) -> str:
    """Return the last few-shot assistant message: right shape, wrong content."""
    for m in reversed(messages):
        if m["role"] == "assistant":
            return m["content"]
    return "0"


class PerfectMemorizer:
    """Answers from the dataset itself: an upper bound on any memorization signal."""

    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.rows = utils.load_csv_rows(csv_file)
        self.text = utils.load_csv_string(csv_file)
        self.df = utils.load_csv_df(csv_file)
        self.feature_names = utils.get_feature_names(csv_file)
        # prefix (any number of consecutive rows) -> next row
        self.next_row = {}
        for i in range(len(self.rows) - 1):
            for k in (8, 10):
                if i + k < len(self.rows):
                    self.next_row["\n".join(self.rows[i:i + k])] = self.rows[i + k]

    def __call__(self, messages: List[Dict]) -> str:
        prompt = messages[-1]["content"]

        # row completion / first token: exact block lookup
        if prompt in self.next_row:
            return self.next_row[prompt]

        # header test: the prompt is a raw prefix of the CSV file
        if self.text.startswith(prompt[:200]) or prompt in self.text:
            idx = self.text.find(prompt)
            if idx == 0 or idx > 0:
                return self.text[idx + len(prompt): idx + len(prompt) + 500]

        # feature completion: "f1 = v1, f2 = v2, ..." -> "target = value"
        if " = " in prompt:
            return self._complete_feature(prompt, messages)

        # fall back to the row after the last complete line seen
        tail = prompt.strip().split("\n")[-1]
        for i, row in enumerate(self.rows[:-1]):
            if row == tail:
                return self.rows[i + 1]
        return ""

    def _complete_feature(self, prompt: str, messages: List[Dict]) -> str:
        # the target feature is whatever the few-shot answers are about
        target = None
        for m in reversed(messages):
            if m["role"] == "assistant" and " = " in m["content"]:
                target = m["content"].split(" = ")[0].strip()
                break
        if target is None or target not in self.df.columns:
            return ""

        # parse conditioning values and find the matching row
        conds = {}
        for part in prompt.split(", "):
            if " = " in part:
                name, value = part.split(" = ", 1)
                conds[name.strip()] = value.strip()

        mask = None
        for name, value in conds.items():
            if name not in self.df.columns:
                continue
            col = self.df[name].astype(str).str.strip()
            m = col == value
            mask = m if mask is None else (mask & m)
        if mask is None or not mask.any():
            return ""
        value = self.df.loc[mask, target].iloc[0]
        return f"{target} = {value}"
