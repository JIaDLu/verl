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
"""Three metrics for scoring a finished CollabLLM trajectory.

Each ``score_*`` function takes a TrajectoryEntry plus task context and
returns a single float. They are pure (no shared state) so they can be
fanned out to a ThreadPoolExecutor without locking concerns.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .config import CollabLLMConfig
from .llm_client import LLMClient
from .prompts import (
    render_accuracy_judge_prompt,
    render_interactivity_judge_prompt,
    safe_parse_json,
)
from .trajectory import TrajectoryEntry

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# tiktoken — loaded lazily and cached, since constructing an encoding
# is non-trivial and we'll call it thousands of times per step.
# ----------------------------------------------------------------------
_TIKTOKEN_CACHE: dict[str, Any] = {}
_TIKTOKEN_LOCK = threading.Lock()


def _get_encoding(name: str):
    enc = _TIKTOKEN_CACHE.get(name)
    if enc is not None:
        return enc
    with _TIKTOKEN_LOCK:
        enc = _TIKTOKEN_CACHE.get(name)
        if enc is not None:
            return enc
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "tiktoken required for token_amount metric. Install with: pip install tiktoken"
            ) from e
        enc = tiktoken.get_encoding(name)
        _TIKTOKEN_CACHE[name] = enc
        return enc


def score_accuracy(
    entry: TrajectoryEntry,
    *,
    single_turn_prompt: str,
    ground_truth: str,
    config: CollabLLMConfig,
    judge_client: LLMClient,
) -> float:
    """Use an LLM judge to decide if the assistant's final reply is correct.

    Returns 0.0 or 1.0. On any failure (network, parse) returns
    ``config.accuracy_default`` and logs a warning so a single bad call
    never poisons the whole training step.
    """
    completion = entry.last_assistant()
    if not completion:
        # Conversation never produced an assistant turn — can't be correct.
        return float(config.accuracy_default)

    prompt = render_accuracy_judge_prompt(
        single_turn_prompt=single_turn_prompt,
        ground_truth=ground_truth,
        completion=completion,
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        raw = judge_client.chat(
            messages=messages,
            model=config.llm_model,
            temperature=config.judge_temperature,
            max_tokens=config.judge_max_tokens,
            json_mode=True,
            retries=config.api_retries,
            initial_backoff=config.api_initial_backoff,
            tag="accuracy_judge",
            meta={"origin_id": entry.origin_id, "branch_id": entry.branch_id},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Accuracy judge failed: %s", e)
        return float(config.accuracy_default)

    parsed = safe_parse_json(raw, default=None)
    if not isinstance(parsed, dict) or "accuracy" not in parsed:
        logger.warning("Accuracy judge returned malformed output: %r", raw[:200])
        return float(config.accuracy_default)

    val = parsed["accuracy"]
    try:
        score = float(val)
    except (TypeError, ValueError):
        logger.warning("Accuracy judge: non-numeric value %r", val)
        return float(config.accuracy_default)
    # Clamp to {0, 1} — paper defines binary accuracy.
    return 1.0 if score >= 0.5 else 0.0


def score_token_amount(
    entry: TrajectoryEntry,
    *,
    config: CollabLLMConfig,
) -> float:
    """Count assistant tokens in the trajectory; return *k-tokens* (count / 1000).

    The penalty weight (typically negative) is applied later by the
    aggregator. We clip the *raw* count at ``token_amount_clip_k`` k-tokens
    so this single metric can't dominate. Pure local, no API call.
    """
    encoding = _get_encoding(config.tiktoken_encoding)
    total = 0
    for msg in entry.conversation:
        if msg.get("role") == "assistant":
            total += len(encoding.encode(msg.get("content", "")))
    k = total / 1000.0
    if k > config.token_amount_clip_k:
        k = config.token_amount_clip_k
    return k


def score_interactivity(
    entry: TrajectoryEntry,
    *,
    config: CollabLLMConfig,
    judge_client: LLMClient,
) -> float:
    """Use an LLM judge to score interaction quality in [0, 1]."""
    if not entry.conversation:
        return float(config.interactivity_default)

    prompt = render_interactivity_judge_prompt(entry.conversation)
    messages = [{"role": "user", "content": prompt}]
    try:
        raw = judge_client.chat(
            messages=messages,
            model=config.llm_model,
            temperature=config.judge_temperature,
            max_tokens=config.judge_max_tokens,
            json_mode=True,
            retries=config.api_retries,
            initial_backoff=config.api_initial_backoff,
            tag="interactivity_judge",
            meta={"origin_id": entry.origin_id, "branch_id": entry.branch_id},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Interactivity judge failed: %s", e)
        return float(config.interactivity_default)

    parsed = safe_parse_json(raw, default=None)
    if not isinstance(parsed, dict) or "interactivity" not in parsed:
        logger.warning("Interactivity judge malformed: %r", raw[:200])
        return float(config.interactivity_default)

    try:
        score = float(parsed["interactivity"])
    except (TypeError, ValueError):
        return float(config.interactivity_default)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


# ----------------------------------------------------------------------
# Dispatcher: maps metric name -> scorer call.
# Used by the parallel scorer in forward_sampling.py.
# ----------------------------------------------------------------------
def score_one(
    metric_name: str,
    entry: TrajectoryEntry,
    *,
    single_turn_prompt: str,
    ground_truth: str,
    config: CollabLLMConfig,
    judge_client: LLMClient,
) -> float:
    if metric_name == "accuracy":
        return score_accuracy(
            entry,
            single_turn_prompt=single_turn_prompt,
            ground_truth=ground_truth,
            config=config,
            judge_client=judge_client,
        )
    if metric_name == "token_amount":
        return score_token_amount(entry, config=config)
    if metric_name == "interactivity":
        return score_interactivity(entry, config=config, judge_client=judge_client)
    raise ValueError(f"Unknown metric: {metric_name}")
