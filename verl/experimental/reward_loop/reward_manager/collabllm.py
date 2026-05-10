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

Two execution modes — both selected automatically without code changes
to verl's framework:

  Inline mode (production)
    Used by ``recipes.collabllm.main_collabllm_ppo``. The recipe's
    trainer subclass injects a ``generation_fn`` that proxies forward
    sampling to the *live* actor's vLLM (zero checkpoint drift).

    Activated when ``generation_fn`` is passed to ``__init__`` AND the
    ``NEEDS_INLINE_DISPATCH`` class flag is honored by the dispatcher.
    The recipe's dispatcher calls ``run_batch`` directly, in-process.

  HTTP mode (standalone testing only)
    Used by ``tests/collabllm/test_pipeline.py`` and any non-recipe
    invocation that lands here via the regular Ray reward dispatch.
    Forward sampling is served by a separate vLLM HTTP endpoint
    (started by ``examples/grpo_trainer/start_reward_vllm.sh``).

    Activated by default when no ``generation_fn`` is provided.

The class flag ``NEEDS_INLINE_DISPATCH = True`` is a hint to recipe
trainers — not a verl framework contract. The default verl reward loop
ignores it; it only matters when a recipe explicitly checks for it.

Config: all knobs come from ``config.reward.reward_kwargs``. See
:class:`CollabLLMConfig` for the full schema.

Registered name: ``"collabllm"``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score.collabllm.config import CollabLLMConfig
from verl.utils.reward_score.collabllm.forward_sampling import compute_multiturn_rewards
from verl.utils.reward_score.collabllm.llm_client import LLMClient, resolve_api_key
from verl.utils.reward_score.collabllm.policy_caller import (
    GenFnPolicyCaller,
    HTTPPolicyCaller,
    PolicyCaller,
)

logger = logging.getLogger(__name__)


@register("collabllm")
class CollabLLMRewardManager(RewardManagerBase):
    """Multi-turn aware reward manager (CollabLLM).

    Behavior is parameterized by:

    * ``generation_fn`` (kwarg, optional): if provided, forward sampling
      uses this callable instead of an HTTP vLLM. The callable contract
      is ``list[messages] -> list[reply]``. Provided by the recipe
      trainer; ``None`` in standalone test mode.
    * ``config.reward.reward_kwargs.policy_api_base`` (yaml): used only
      when ``generation_fn`` is None — the HTTP fallback endpoint.

    Class-level state is initialized once per process via :meth:`init_class`
    (LLM client + parsed config). Per-instance state holds the
    PolicyCaller, which depends on whether ``generation_fn`` was passed.
    """

    # Recipe trainer hook: when set, recipe-level dispatcher is expected
    # to bypass the default Ray-based reward loop and call this manager
    # in-process so generation_fn (which captures actor_rollout_wg) works.
    NEEDS_INLINE_DISPATCH = True

    # Class-level shared state.
    _cfg: CollabLLMConfig | None = None
    _sim_client: LLMClient | None = None
    _judge_client: LLMClient | None = None
    _policy_client: LLMClient | None = None  # used only by HTTP fallback

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        *,
        generation_fn: Callable[[list[list[dict[str, str]]]], list[str]] | None = None,
        **reward_kwargs: Any,
    ):
        super().__init__(config, tokenizer, compute_score)
        # ``compute_score`` is unused for CollabLLM — we have our own pipeline.
        self._reward_kwargs = reward_kwargs
        self._generation_fn = generation_fn
        self._policy_caller: PolicyCaller | None = None
        # Build the PolicyCaller after init_class runs (need _cfg).
        # We construct lazily on first use to keep __init__ side-effect-free.

    @classmethod
    def init_class(cls, config, tokenizer):
        if cls._class_initialized:
            return
        cls._class_initialized = True

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

        api_key = resolve_api_key(cfg.llm_api_key_env, fallback="EMPTY")
        cls._sim_client = LLMClient.get_shared(
            base_url=cfg.llm_api_base,
            api_key=api_key,
            request_timeout=cfg.api_request_timeout,
            trace_path=cfg.trace_path,
        )
        cls._judge_client = cls._sim_client

        # The HTTP policy client is only used as a fallback. We always
        # construct it so standalone-test setups keep working — but the
        # production path (with generation_fn) bypasses it.
        cls._policy_client = LLMClient.get_shared(
            base_url=cfg.policy_api_base,
            api_key=cfg.policy_api_key,
            request_timeout=cfg.api_request_timeout,
            trace_path=cfg.trace_path,
        )

        logger.info(
            "[CollabLLMRewardManager] initialized: model=%s, policy_endpoint=%s, "
            "metrics=%s, weights=%s, window=%d, branches=%d, trace=%s",
            cfg.llm_model, cfg.policy_api_base,
            cfg.metric_names, cfg.metric_weights,
            cfg.forward_sampling_window, cfg.forward_sampling_branches,
            cfg.trace_path or "off",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_policy_caller(self) -> PolicyCaller:
        """Build the PolicyCaller on first use."""
        if self._policy_caller is not None:
            return self._policy_caller
        cfg = self._cfg
        assert cfg is not None, "init_class must run before use"
        if self._generation_fn is not None:
            self._policy_caller = GenFnPolicyCaller(
                gen_fn=self._generation_fn,
                trace_writer=self._make_trace_writer(),
            )
            logger.info("CollabLLM using GenFnPolicyCaller (live actor, zero drift)")
        else:
            assert self._policy_client is not None
            self._policy_caller = HTTPPolicyCaller(
                client=self._policy_client,
                model=cfg.policy_model,
                temperature=cfg.policy_temperature,
                max_tokens=cfg.policy_max_tokens,
                top_p=cfg.policy_top_p,
                retries=cfg.api_retries,
                initial_backoff=cfg.api_initial_backoff,
                max_workers=cfg.max_policy_workers,
            )
            logger.info("CollabLLM using HTTPPolicyCaller (fallback, drifts)")
        return self._policy_caller

    def _make_trace_writer(self):
        """Build a callable the GenFn caller can use to mirror its calls
        into the same JSONL trace as the HTTP path. Returns None if
        tracing is off."""
        cfg = self._cfg
        if cfg is None or not cfg.trace_path:
            return None

        # Reuse the LLMClient's trace machinery for parity with HTTP path.
        trace_client = self._policy_client  # has the file lock + write path
        if trace_client is None:
            return None

        def _writer(messages: list[dict], reply: str, meta: dict | None) -> None:
            trace_client._record_trace(
                tag="policy",
                meta=meta or {},
                request_kwargs={"model": cfg.policy_model, "messages": messages,
                                "temperature": cfg.policy_temperature},
                response_text=reply,
                latency_ms=0.0,  # gen_fn doesn't measure per-call; bulk batch
            )
        return _writer

    def _decode_response(self, data_item) -> str:
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:int(valid_response_length)]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    @staticmethod
    def _extract_prompt_messages(data_item) -> list[dict[str, str]]:
        nt = data_item.non_tensor_batch
        for key in ("raw_prompt", "prompt"):
            messages = nt.get(key, None)
            if messages is None:
                continue
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            if not isinstance(messages, list) or not messages:
                continue
            if all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
                return [{"role": m["role"], "content": m["content"]} for m in messages]
        return [{"role": "user", "content": ""}]

    def _extract_record(self, data_item) -> tuple[list[dict[str, str]], str, str, str]:
        """Pull (prompt_msgs, response_str, single_turn_prompt, ground_truth) from one item."""
        rm = data_item.non_tensor_batch.get("reward_model", {})
        ground_truth = str(rm.get("ground_truth", ""))
        single_turn_prompt = str(rm.get("single_turn_prompt", "") or "")
        prompt_msgs = self._extract_prompt_messages(data_item)
        if not single_turn_prompt:
            for m in prompt_msgs:
                if m.get("role") == "user":
                    single_turn_prompt = m.get("content", "")
                    break
        response_str = self._decode_response(data_item)
        return prompt_msgs, response_str, single_turn_prompt, ground_truth

    # ------------------------------------------------------------------
    # Per-sample async entry (verl's default Ray dispatch path).
    # ------------------------------------------------------------------
    async def run_single(self, data: DataProto) -> dict:
        """Process a single batch item.

        This path is used when the default verl reward loop dispatches
        via Ray remote workers. It works correctly only with the HTTP
        PolicyCaller (the gen_fn callable cannot cross Ray process
        boundaries cleanly). The recipe path uses ``run_batch`` instead.
        """
        assert len(data) == 1
        data_item = data[0]

        prompt_msgs, response_str, stp, gt = await self.loop.run_in_executor(
            None, self._extract_record, data_item,
        )

        cfg = self._cfg
        sim_client = self._sim_client
        judge_client = self._judge_client
        policy_caller = self._ensure_policy_caller()
        assert cfg is not None and sim_client and judge_client

        def _run() -> tuple[list[float], list[dict]]:
            return compute_multiturn_rewards(
                rollout_pairs=[(prompt_msgs, response_str)],
                single_turn_prompts=[stp],
                ground_truths=[gt],
                config=cfg,
                sim_client=sim_client,
                policy_caller=policy_caller,
                judge_client=judge_client,
            )

        try:
            mr_values, debug_info = await self.loop.run_in_executor(None, _run)
        except Exception as e:  # noqa: BLE001
            logger.exception("CollabLLM reward pipeline failed; defaulting to 0.0: %s", e)
            return {"reward_score": 0.0, "reward_extra_info": {"error": str(e)[:200]}}

        return self._build_result(mr_values[0], debug_info[0])

    # ------------------------------------------------------------------
    # Whole-batch entry (recipe path).
    # ------------------------------------------------------------------
    def run_batch(self, batch: DataProto) -> list[dict]:
        """Process the entire batch in one pool.

        Called by the recipe's inline dispatcher. Builds one big forward
        sampling pool of ``len(batch) * branches`` entries and submits a
        SINGLE batched policy call per turn, which lets ``GenFnPolicyCaller``
        send everything to the live actor's vLLM in one shot.

        Returns:
            One ``{"reward_score": ..., "reward_extra_info": ...}`` dict
            per batch item, in original order.
        """
        n = len(batch)
        rollout_pairs: list[tuple[list[dict[str, str]], str]] = []
        single_turn_prompts: list[str] = []
        ground_truths: list[str] = []
        for i in range(n):
            prompt_msgs, response_str, stp, gt = self._extract_record(batch[i])
            rollout_pairs.append((prompt_msgs, response_str))
            single_turn_prompts.append(stp)
            ground_truths.append(gt)

        cfg = self._cfg
        sim_client = self._sim_client
        judge_client = self._judge_client
        policy_caller = self._ensure_policy_caller()
        assert cfg is not None and sim_client and judge_client

        try:
            mr_values, debug_info = compute_multiturn_rewards(
                rollout_pairs=rollout_pairs,
                single_turn_prompts=single_turn_prompts,
                ground_truths=ground_truths,
                config=cfg,
                sim_client=sim_client,
                policy_caller=policy_caller,
                judge_client=judge_client,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("CollabLLM run_batch failed; zeroing rewards: %s", e)
            return [{"reward_score": 0.0, "reward_extra_info": {"error": str(e)[:200]}}
                    for _ in range(n)]

        return [self._build_result(mr_values[i], debug_info[i]) for i in range(n)]

    # ------------------------------------------------------------------
    # Result formatting (shared by run_single and run_batch)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_result(mr: float, info: dict) -> dict:
        extra: dict[str, Any] = {"mr": float(mr), "branches_completed": info.get("branches", 0)}
        for m, v in info.get("metric_avg", {}).items():
            extra[f"metric/{m}"] = float(v)
        terminal_reasons = info.get("terminal_reasons", [])
        if terminal_reasons:
            extra["terminal_reason_majority"] = max(set(terminal_reasons), key=terminal_reasons.count)
        return {"reward_score": float(mr), "reward_extra_info": extra}
