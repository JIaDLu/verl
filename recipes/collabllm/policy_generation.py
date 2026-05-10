# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build a ``generation_fn`` that proxies forward sampling to the live
actor's vLLM via verl's ``async_rollout_manager``.

Lives in the recipe — *not* in verl framework code — because it
references trainer-internal handles. The reward manager doesn't need
to know any of this; it just calls ``gen_fn(messages_batch)`` and gets
back a list of strings.

The wrapper:
  1. Renders each conversation through the tokenizer's chat template
     (``add_generation_prompt=True`` so the assistant turn is open).
  2. Tokenizes with left padding to ``max_prompt_length``.
  3. Wraps into a ``DataProto`` matching what
     ``async_rollout_manager.generate_sequences`` expects.
  4. Calls generate_sequences (single batched call).
  5. Decodes each sample's response token IDs back to a string.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

logger = logging.getLogger(__name__)


def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Cumulative position IDs from a left-padded attention mask."""
    # For left-padding, position 0 is the first non-pad token.
    pos = attention_mask.long().cumsum(dim=-1) - 1
    pos = pos.clamp(min=0)
    return pos


def make_generation_fn(
    *,
    async_rollout_manager,
    tokenizer,
    max_prompt_length: int,
    max_response_length: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    extra_meta_info: dict | None = None,
) -> Callable[[list[list[dict[str, str]]]], list[str]]:
    """Construct the generation callable wired into ``GenFnPolicyCaller``.

    Args:
        async_rollout_manager: ``self.async_rollout_manager`` from the
            trainer (the live AgentLoopManager wrapping vLLM).
        tokenizer: same tokenizer the trainer uses; chat-template aware.
        max_prompt_length / max_response_length: must match the trainer's
            data config so the rollout manager's batching assumptions
            stay consistent.
        temperature / top_p: forward sampling generation knobs (passed
            through ``meta_info`` so the rollout manager can override its
            defaults for our calls).
        extra_meta_info: optional dict merged into ``meta_info`` for
            this call (e.g. tracing flags).

    Returns:
        ``gen_fn(messages_batch) -> replies``.
    """

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id

    def gen_fn(messages_batch: list[list[dict[str, str]]]) -> list[str]:
        n = len(messages_batch)
        if n == 0:
            return []

        # 1. Render through chat template.
        prompts: list[str] = []
        for msgs in messages_batch:
            prompts.append(
                tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False,
                )
            )

        # 2. Tokenize with left padding.
        prev_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            enc = tokenizer(
                prompts,
                padding="max_length",
                max_length=max_prompt_length,
                truncation=True,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = prev_padding_side

        input_ids: torch.Tensor = enc["input_ids"]
        attention_mask: torch.Tensor = enc["attention_mask"]
        position_ids = _build_position_ids(attention_mask)

        # 3. Build the DataProto.
        td = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=n,
        )
        gen_batch = DataProto(batch=td, non_tensor_batch={}, meta_info={
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_response_length": max_response_length,
            "eos_token_id": [eos_token_id] if eos_token_id is not None else None,
            "pad_token_id": pad_token_id,
            "recompute_log_prob": False,
            **(extra_meta_info or {}),
        })

        # 4. Single batched call into the live actor's vLLM.
        try:
            out = async_rollout_manager.generate_sequences(gen_batch)
        except Exception as e:  # noqa: BLE001
            logger.error("generate_sequences failed for n=%d: %s", n, e)
            return [""] * n

        # 5. Decode responses back to strings.
        responses = out.batch.get("responses", None)
        if responses is None:
            logger.error("generate_sequences output missing 'responses' key")
            return [""] * n

        # The output's attention_mask covers prompt+response; the
        # response section starts right after the prompt.
        full_mask = out.batch.get("attention_mask")
        if full_mask is None:
            # Fallback: assume full response length is valid.
            response_lengths = torch.full((n,), responses.shape[-1], dtype=torch.long)
        else:
            prompt_len = input_ids.shape[-1]
            response_lengths = full_mask[:, prompt_len:].sum(dim=-1).long()

        results: list[str] = []
        for i in range(n):
            length = int(response_lengths[i].item()) if i < len(response_lengths) else responses.shape[-1]
            ids = responses[i, :length].tolist() if length > 0 else []
            text = tokenizer.decode(ids, skip_special_tokens=True) if ids else ""
            results.append(text)
        return results

    return gen_fn
