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
"""Hydra entry point for CollabLLM GRPO training.

Mirrors ``verl.trainer.main_ppo.main`` but routes through
:class:`CollabLLMTaskRunner`, which instantiates a trainer that uses
the live actor's vLLM for forward sampling. Reuses verl's default
``ppo_trainer`` config tree — no separate config file needed.

Run:
    python -m recipes.collabllm.main_collabllm_ppo \\
        algorithm.adv_estimator=grpo \\
        actor_rollout_ref.model.path=/path/to/sft_merged \\
        ...
"""

from __future__ import annotations

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

from .task_runner import CollabLLMTaskRunner


@hydra.main(
    config_path="../../verl/trainer/config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    # Wrap our subclass as a Ray remote actor — same pattern as default.
    task_runner_class = ray.remote(num_cpus=1)(CollabLLMTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
