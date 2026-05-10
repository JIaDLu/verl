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
"""Config dataclass for CollabLLM multi-turn aware reward.

All knobs come from a single config object so they can be passed via
verl's ``custom_reward_function.reward_kwargs`` or the reward manager's
``reward_kwargs`` field — no hard-coded values inside the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_SUPPORTED_METRICS = {"accuracy", "token_amount", "interactivity"}


@dataclass
class CollabLLMConfig:
    # ---------- Forward sampling ----------
    forward_sampling_window: int = 2          # w turns to simulate ahead
    forward_sampling_branches: int = 3        # how many independent futures per response
    max_seq_len: int = 4096                   # token-budget guard for whole conversation
    terminal_signal: str = "[TERMINATE]"      # printed by user simulator when goal reached
    task_desc: str = "math problem solving"   # filled into user simulator prompt

    # ---------- Metrics ----------
    metric_names: tuple[str, ...] = ("accuracy", "token_amount", "interactivity")
    metric_weights: tuple[float, ...] = (1.0, -0.5, 1.0)
    token_amount_clip_k: float = 4.0          # upper bound on penalty contribution (k tokens)
    tiktoken_encoding: str = "cl100k_base"    # offline token counter

    # ---------- LLM API for User Simulator + Judges ----------
    # Defaults target DeepSeek-v4-pro via its OpenAI-compatible endpoint;
    # override via reward_kwargs to switch providers.
    llm_api_base: str = "https://api.deepseek.com"
    llm_api_key_env: str = "DEEPSEEK_API_KEY"   # name of env var holding the key
    llm_model: str = "deepseek-v4-pro"
    user_simulator_temperature: float = 0.8
    user_simulator_max_tokens: int = 512
    judge_temperature: float = 0.0
    judge_max_tokens: int = 512

    # ---------- vLLM for Policy Forward Generation ----------
    # OpenAI-compatible endpoint exposed by a vLLM server holding the *current* policy
    # (or the SFT-merged checkpoint as an approximation, see launch notes).
    policy_api_base: str = "http://127.0.0.1:8000/v1"
    policy_api_key: str = "EMPTY"
    policy_model: str = "collabllm-policy"
    policy_temperature: float = 1.0
    policy_top_p: float = 0.95
    policy_max_tokens: int = 512

    # ---------- Concurrency & robustness ----------
    max_metric_workers: int = 64              # ThreadPool size for accuracy + interactivity
    max_simulator_workers: int = 64           # ThreadPool size for user-simulator calls
    max_policy_workers: int = 64              # ThreadPool size for vLLM policy calls
    api_retries: int = 3
    api_initial_backoff: float = 1.0          # seconds; doubles each retry
    api_request_timeout: float = 60.0

    # ---------- Defaults on parse/network failure ----------
    accuracy_default: float = 0.0
    interactivity_default: float = 0.0

    # ---------- Tracing (debug / audit) ----------
    # If set, every LLM call (User Sim / Judges / Policy) appends one
    # JSONL line to this file. Disabled in production by default.
    trace_path: str | None = None

    def __post_init__(self) -> None:
        if len(self.metric_names) != len(self.metric_weights):
            raise ValueError(
                f"metric_names ({len(self.metric_names)}) and metric_weights "
                f"({len(self.metric_weights)}) must have equal length"
            )
        unknown = set(self.metric_names) - _SUPPORTED_METRICS
        if unknown:
            raise ValueError(f"Unknown metric(s): {unknown}. Supported: {_SUPPORTED_METRICS}")
        if self.forward_sampling_window < 1:
            raise ValueError("forward_sampling_window must be >= 1")
        if self.forward_sampling_branches < 1:
            raise ValueError("forward_sampling_branches must be >= 1")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "CollabLLMConfig":
        """Build a config from a flat dict, ignoring unknown keys with a warning.

        verl passes ``reward_kwargs`` as a dict; this lets the user override only
        what they care about without copying the full schema.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {}
        unknown = []
        for k, v in kwargs.items():
            if k in valid_fields:
                clean[k] = v
            else:
                unknown.append(k)
        if unknown:
            import warnings
            warnings.warn(f"CollabLLMConfig: ignoring unknown keys: {unknown}", stacklevel=2)
        # tuples come over yaml as lists; coerce back
        for k in ("metric_names", "metric_weights"):
            if k in clean and isinstance(clean[k], list):
                clean[k] = tuple(clean[k])
        return cls(**clean)
