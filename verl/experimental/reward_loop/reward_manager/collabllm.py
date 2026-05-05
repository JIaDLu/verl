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
"""CollabLLM reward manager for verl.

Per-response evaluation: ``run_single`` builds a 1×B trajectory pool
(B = ``forward_sampling_branches``), simulates forward, scores each
branch with the configured metrics, averages branches into MR, and
returns it as the reward.

Cross-response concurrency comes from verl itself: it dispatches every
batch item to ``run_single`` concurrently via ``asyncio.gather``, so
the LLM API calls from many responses overlap naturally. Within one
response we still parallelize across branches and metrics.

Config: all knobs come from ``config.reward.reward_kwargs`` — no
hard-coded values. See :class:`CollabLLMConfig` for the schema.

Registered name: ``"collabllm"``. Use it in your launch script as
``reward.reward_manager.name=collabllm``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score.collabllm.config import CollabLLMConfig
from verl.utils.reward_score.collabllm.forward_sampling import compute_multiturn_rewards
from verl.utils.reward_score.collabllm.llm_client import LLMClient, resolve_api_key

logger = logging.getLogger(__name__)


@register("collabllm")
class CollabLLMRewardManager(RewardManagerBase):
    """Multi-turn aware reward manager (CollabLLM).

    Class-level state (LLM clients, parsed config) is initialized once
    per process via :meth:`init_class`. Per-instance state is kept
    minimal so the reward manager can be safely cloned by Ray.
    """

    # Class-level shared state — populated by init_class.
    _cfg: CollabLLMConfig | None = None
    _sim_client: LLMClient | None = None
    _judge_client: LLMClient | None = None
    _policy_client: LLMClient | None = None

    def __init__(self, config, tokenizer, compute_score, **reward_kwargs: Any):
        super().__init__(config, tokenizer, compute_score)
        # ``compute_score`` is unused for CollabLLM — we have our own pipeline.
        # We still accept it so the loader signature matches.
        self._reward_kwargs = reward_kwargs

    @classmethod
    def init_class(cls, config, tokenizer):
        """Build the parsed config + shared LLM clients exactly once per process."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

        # ``reward_kwargs`` is a flat dict on the reward config. We accept
        # it via the trainer's ``custom_reward_function.reward_kwargs`` OR
        # ``reward.reward_manager.reward_kwargs`` — try both.
        rk: dict[str, Any] = {}
        try:
            rk_cfg = config.reward.get("reward_kwargs", None)
            if rk_cfg is not None:
                rk.update(dict(rk_cfg))
        except Exception:  # noqa: BLE001
            pass
        try:
            crf = config.reward.get("custom_reward_function", None)
            if crf is not None and crf.get("reward_kwargs", None) is not None:
                rk.update(dict(crf.reward_kwargs))
        except Exception:  # noqa: BLE001
            pass

        cfg = CollabLLMConfig.from_kwargs(**rk)
        cls._cfg = cfg

        # User Simulator and Judge share the same provider (and may share
        # an API key) → reuse one client. Policy uses the local vLLM
        # endpoint with a different base_url.
        api_key = resolve_api_key(cfg.llm_api_key_env, fallback="EMPTY")
        cls._sim_client = LLMClient.get_shared(
            base_url=cfg.llm_api_base,
            api_key=api_key,
            request_timeout=cfg.api_request_timeout,
        )
        cls._judge_client = cls._sim_client

        cls._policy_client = LLMClient.get_shared(
            base_url=cfg.policy_api_base,
            api_key=cfg.policy_api_key,
            request_timeout=cfg.api_request_timeout,
        )

        logger.info(
            "[CollabLLMRewardManager] initialized: model=%s, policy=%s, "
            "metrics=%s, weights=%s, window=%d, branches=%d",
            cfg.llm_model, cfg.policy_model,
            cfg.metric_names, cfg.metric_weights,
            cfg.forward_sampling_window, cfg.forward_sampling_branches,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _decode_response(self, data_item) -> str:
        """Decode the assistant response token IDs to a clean string."""
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:int(valid_response_length)]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    @staticmethod
    def _extract_prompt_messages(data_item) -> list[dict[str, str]]:
        """Get the prompt as a list of {role, content} chat messages.

        verl typically stores the original messages in
        ``non_tensor_batch["raw_prompt"]`` (preferred) or ``["prompt"]``.
        Fall back to a single-turn user wrapper if neither exists.
        """
        nt = data_item.non_tensor_batch
        for key in ("raw_prompt", "prompt"):
            messages = nt.get(key, None)
            if messages is None:
                continue
            # numpy arrays of dicts come back as 0-d arrays sometimes
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            if not isinstance(messages, list) or not messages:
                continue
            # Validate shape
            if all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
                return [{"role": m["role"], "content": m["content"]} for m in messages]
        # Last-resort fallback: nothing usable found.
        return [{"role": "user", "content": ""}]

    # ------------------------------------------------------------------
    # Main per-sample entry point (verl async contract)
    # ------------------------------------------------------------------
    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "CollabLLMRewardManager.run_single expects a single item"
        data_item = data[0]

        # Pull task context.
        rm = data_item.non_tensor_batch.get("reward_model", {})
        ground_truth = str(rm.get("ground_truth", ""))
        single_turn_prompt = str(rm.get("single_turn_prompt", "") or "")
        if not single_turn_prompt:
            # Fall back to the raw user prompt if the dataset lacks it.
            messages = self._extract_prompt_messages(data_item)
            for m in messages:
                if m.get("role") == "user":
                    single_turn_prompt = m.get("content", "")
                    break

        # Decode response text + reconstruct prompt history.
        # tokenizer.decode is sync and CPU-bound; offload to executor so
        # we don't block the event loop.
        response_str = await self.loop.run_in_executor(None, self._decode_response, data_item)
        prompt_msgs = await self.loop.run_in_executor(
            None, self._extract_prompt_messages, data_item,
        )

        # Run the (sync) forward-sampling pipeline in a thread so the
        # outer event loop can serve other run_single calls in parallel.
        cfg = self._cfg
        sim_client = self._sim_client
        judge_client = self._judge_client
        policy_client = self._policy_client
        assert cfg is not None and sim_client and judge_client and policy_client

        def _run() -> tuple[list[float], list[dict]]:
            return compute_multiturn_rewards(
                rollout_pairs=[(prompt_msgs, response_str)],
                single_turn_prompts=[single_turn_prompt],
                ground_truths=[ground_truth],
                config=cfg,
                sim_client=sim_client,
                policy_client=policy_client,
                judge_client=judge_client,
            )

        try:
            mr_values, debug_info = await self.loop.run_in_executor(None, _run)
        except Exception as e:  # noqa: BLE001
            logger.exception("CollabLLM reward pipeline failed; defaulting to 0.0: %s", e)
            return {"reward_score": 0.0, "reward_extra_info": {"error": str(e)[:200]}}

        mr = float(mr_values[0])
        info = debug_info[0]

        # Surface a few inspection-friendly fields for wandb / logs.
        extra: dict[str, Any] = {"mr": mr, "branches_completed": info.get("branches", 0)}
        for m, v in info.get("metric_avg", {}).items():
            extra[f"metric/{m}"] = float(v)
        terminal_reasons = info.get("terminal_reasons", [])
        if terminal_reasons:
            extra["terminal_reason_majority"] = max(set(terminal_reasons), key=terminal_reasons.count)

        return {"reward_score": mr, "reward_extra_info": extra}
