"""HuggingFace backend for tabmemcheck (stage 1 of the adaptation plan).

Implements tabmemcheck.LLM_Interface for local HF models so that all
memorization tests run without any OpenAI/Anthropic API. Test logic and
success criteria in tabmemcheck are untouched — this is a backend only.

Design constraints (see PREREGISTRATION.md §2, §3):
- temperature 0 maps to greedy decoding (deterministic on fixed hardware);
- model revisions are pinned by the caller (models.lock);
- chat templates without a system role get the system message merged into
  the first user turn (several Russian models need this fallback).
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tabmemcheck


@dataclass
class HFLLM(tabmemcheck.LLM_Interface):
    model_name: str = ""
    revision: Optional[str] = None
    device: str = "cpu"
    dtype: Optional[torch.dtype] = None
    chat_mode: bool = True
    quantization_config: object = None

    model: object = field(default=None, repr=False)
    tokenizer: object = field(default=None, repr=False)

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision
        )
        kwargs = {"revision": self.revision,
                  "torch_dtype": self.dtype if self.dtype is not None else "auto"}
        if self.quantization_config is not None:
            # accelerate places a quantized model itself; moving it afterwards fails
            kwargs.update(quantization_config=self.quantization_config,
                          device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if self.quantization_config is None:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _generate(self, input_ids, temperature: float, max_tokens: int) -> str:
        do_sample = temperature > 0
        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def completion(self, prompt, temperature, max_tokens):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )
        return self._generate(input_ids, temperature, max_tokens)

    def chat_completion(self, messages, temperature, max_tokens):
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
        except Exception:
            # chat template rejects the system role: merge it into the first
            # user message and retry
            merged = []
            system = ""
            for m in messages:
                if m["role"] == "system":
                    system = m["content"]
                elif m["role"] == "user" and system:
                    merged.append(
                        {"role": "user", "content": system + "\n\n" + m["content"]}
                    )
                    system = ""
                else:
                    merged.append(m)
            input_ids = self.tokenizer.apply_chat_template(
                merged, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
        return self._generate(input_ids, temperature, max_tokens)


def hf_setup(model_name: str, revision: Optional[str] = None,
             device: str = "cpu", chat_mode: bool = True) -> HFLLM:
    return HFLLM(model_name=model_name, revision=revision, device=device,
                 chat_mode=chat_mode)
