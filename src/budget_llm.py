"""Budget-enforcing OpenAI backend for tabmemcheck.

Why not tabmemcheck's own OpenAILLM: it discards the API response object, so
token usage and cost are invisible. This class implements the same interface
with identical call parameters, plus:

- a hard client-side spend cap (raises BudgetExceeded before the call that
  would cross it, never after);
- exact usage accounting from `response.usage`, with a tiktoken pre-estimate
  used for the cap decision;
- a dry-run mode that constructs and counts every prompt without sending it,
  so a run can be priced before any money is spent;
- JSONL logging of every prompt/response pair for the record.

Prices are USD per 1M tokens and are deliberately conservative: if a listed
price is stale the run stops early rather than overspending.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import tiktoken
from openai import OpenAI, BadRequestError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

import tabmemcheck


PRICES_USD_PER_1M = {
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    "gpt-3.5-turbo-16k": (3.00, 4.00),
    "gpt-3.5-turbo-16k-0613": (3.00, 4.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-4-0613": (30.00, 60.00),
    "gpt-4": (30.00, 60.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetedOpenAILLM(tabmemcheck.LLM_Interface):
    model: str = "gpt-3.5-turbo-0125"
    budget_usd: float = 1.0
    dry_run: bool = False
    log_path: Optional[str] = None
    client: object = field(default=None, repr=False)
    # dry-run only: produces a stand-in response so that prompt construction,
    # parsing and counting run exactly as they would live
    mock_fn: Optional[Callable[[list], str]] = field(default=None, repr=False)

    n_calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0
    served_models: dict = field(default_factory=dict)

    def __post_init__(self):
        self.chat_mode = True
        if self.client is None and not self.dry_run:
            self.client = OpenAI()
        self._enc = tiktoken.get_encoding("cl100k_base")
        if self.model not in PRICES_USD_PER_1M:
            raise ValueError(f"no price on file for {self.model}; refusing to spend blind")
        self._price_in, self._price_out = PRICES_USD_PER_1M[self.model]

    def _count_messages(self, messages) -> int:
        # OpenAI chat format overhead: ~4 tokens per message plus priming
        return sum(4 + len(self._enc.encode(m["content"])) for m in messages) + 3

    def _charge(self, n_in: int, n_out: int) -> float:
        return (n_in * self._price_in + n_out * self._price_out) / 1e6

    def _log(self, record: dict):
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @retry(
        retry=retry_if_not_exception_type((BadRequestError, BudgetExceeded)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(7),
        reraise=True,
    )
    def _send(self, messages, temperature, max_tokens):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat_completion(self, messages, temperature, max_tokens):
        est_in = self._count_messages(messages)
        worst_case = self.cost_usd + self._charge(est_in, max_tokens)
        if worst_case > self.budget_usd:
            raise BudgetExceeded(
                f"call would bring worst-case spend to ${worst_case:.3f} "
                f"(cap ${self.budget_usd:.2f}); spent so far ${self.cost_usd:.3f} "
                f"over {self.n_calls} calls"
            )

        if self.dry_run:
            content = self.mock_fn(messages) if self.mock_fn else ""
            est_out = min(max_tokens, len(self._enc.encode(content))) if content else max_tokens
            self.n_calls += 1
            self.in_tokens += est_in
            self.out_tokens += est_out
            self.cost_usd += self._charge(est_in, est_out)
            return content

        t0 = time.time()
        response = self._send(messages, temperature, max_tokens)
        content = response.choices[0].message.content or ""
        usage = response.usage
        n_in = usage.prompt_tokens if usage else est_in
        n_out = usage.completion_tokens if usage else max_tokens

        # Aliases are remapped server-side (gpt-3.5-turbo-16k now serves
        # gpt-3.5-turbo-0125), and billing follows what was actually served,
        # so price the served model when we have a price for it.
        served = response.model
        price_in, price_out = self._price_in, self._price_out
        # longest key first: "gpt-4o-mini-2024-07-18" must not match "gpt-4"
        for known in sorted(PRICES_USD_PER_1M, key=len, reverse=True):
            if served.startswith(known):
                price_in, price_out = PRICES_USD_PER_1M[known]
                break
        self.served_models[served] = self.served_models.get(served, 0) + 1

        self.n_calls += 1
        self.in_tokens += n_in
        self.out_tokens += n_out
        self.cost_usd += (n_in * price_in + n_out * price_out) / 1e6

        self._log({
            "model": response.model,
            "requested_model": self.model,
            "messages": messages,
            "response": content,
            "prompt_tokens": n_in,
            "completion_tokens": n_out,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "latency_s": round(time.time() - t0, 2),
            "cumulative_cost_usd": round(self.cost_usd, 5),
        })
        return content

    def completion(self, prompt, temperature, max_tokens):
        raise NotImplementedError("chat models only; base-model completion not used here")

    def summary(self) -> dict:
        return {
            "model": self.model,
            "served_models": self.served_models,
            "calls": self.n_calls,
            "input_tokens": self.in_tokens,
            "output_tokens": self.out_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "dry_run": self.dry_run,
        }

    def __repr__(self):
        return f"{self.model}(spent=${self.cost_usd:.3f})"
