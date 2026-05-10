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
"""CollabLLMRayPPOTrainer — task-local subclass of verl's RayPPOTrainer.

Customizations (limited to two methods, both additive):

  init_workers
    After the parent fully initializes ``async_rollout_manager`` and
    the rest of the worker stack, build a *second* reward manager that
    has access to the live actor's vLLM via an injected ``generation_fn``.
    The default ``self.reward_loop_manager`` (Ray-distributed) stays
    in place — unused by us, but other reward managers continue to use
    it, and removing it would risk side effects elsewhere.

  _compute_reward_colocate
    Dispatch decision: if our gen-fn-aware reward manager declares
    ``NEEDS_INLINE_DISPATCH``, run reward in-process via ``run_batch``;
    otherwise delegate to the parent (preserves default behavior for
    any non-CollabLLM reward manager that might be configured).

Zero changes to verl framework files.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .policy_generation import make_generation_fn

logger = logging.getLogger(__name__)


class CollabLLMRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer subclass that wires CollabLLM's gen-fn-aware reward path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._collabllm_reward_manager = None  # built in init_workers

    # ------------------------------------------------------------------
    def init_workers(self):
        """Run parent worker init, then build the inline reward manager."""
        super().init_workers()
        try:
            self._collabllm_reward_manager = self._build_collabllm_reward_manager()
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "CollabLLM inline reward manager construction failed; falling back "
                "to default Ray reward dispatch (HTTP path). Error: %s", e,
            )
            self._collabllm_reward_manager = None

    def _build_collabllm_reward_manager(self):
        """Construct a CollabLLM reward manager with a live-actor gen_fn injected."""
        rm_name = self.config.reward.reward_manager.name
        rm_cls = get_reward_manager_cls(rm_name)
        if not getattr(rm_cls, "NEEDS_INLINE_DISPATCH", False):
            logger.info(
                "Reward manager %s does not request inline dispatch; "
                "skipping gen_fn injection.", rm_name,
            )
            return None

        gen_fn = make_generation_fn(
            async_rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
            max_prompt_length=int(self.config.data.max_prompt_length),
            max_response_length=int(self.config.data.max_response_length),
            temperature=float(self.config.actor_rollout_ref.rollout.get("temperature", 1.0)),
            top_p=float(self.config.actor_rollout_ref.rollout.get("top_p", 0.95)),
        )

        return rm_cls(
            config=self.config,
            tokenizer=self.tokenizer,
            compute_score=None,
            generation_fn=gen_fn,
        )

    # ------------------------------------------------------------------
    def _compute_reward_colocate(self, batch: DataProto):
        """Override: route through inline run_batch when applicable.

        Falls back to the parent (Ray-distributed) path for any reward
        manager that didn't opt in via NEEDS_INLINE_DISPATCH.
        """
        rm = self._collabllm_reward_manager
        if rm is None:
            return super()._compute_reward_colocate(batch)
        return self._inline_compute_rm_score(batch, rm)

    def _inline_compute_rm_score(self, batch: DataProto, rm) -> DataProto:
        """In-process reward computation that mirrors RewardLoopManager's
        post-processing (so downstream code sees identical fields).

        We call ``rm.run_batch(batch)`` to get one dict per item, then
        place each scalar reward at the last valid response token —
        exactly the layout ``rm_scores`` is expected to have by the
        rest of the trainer.
        """
        results: list[dict[str, Any]] = rm.run_batch(batch)
        n = len(results)
        if n != len(batch):
            raise RuntimeError(
                f"reward manager returned {n} results for batch of {len(batch)}"
            )

        # 1. Build the rm_scores tensor (same shape/layout as RewardLoopManager).
        prompt_length = batch.batch["prompts"].size(1)
        valid_response_length = batch.batch["attention_mask"][:, prompt_length:].sum(dim=1)
        rm_scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        scores = torch.tensor([float(r["reward_score"]) for r in results], dtype=torch.float32)
        # Place the scalar at index (valid_response_length - 1). Guard zero-length.
        idx = (valid_response_length - 1).clamp(min=0).long()
        rm_scores[torch.arange(n), idx] = scores

        td = TensorDict({"rm_scores": rm_scores}, batch_size=n)

        # 2. Aggregate reward_extra_info into non_tensor_batch.
        extra = [r.get("reward_extra_info", {}) for r in results]
        # Union of keys across items so we don't drop anything if
        # different items emit different metric subsets.
        keys: list[str] = []
        seen = set()
        for d in extra:
            for k in d.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        non_tensor_batch: dict[str, np.ndarray] = {}
        for k in keys:
            vals = [d.get(k, None) for d in extra]
            non_tensor_batch[k] = np.array(vals, dtype=object)

        return DataProto(
            batch=td,
            non_tensor_batch=non_tensor_batch,
            meta_info={"reward_extra_keys": keys},
        )
