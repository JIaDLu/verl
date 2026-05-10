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
"""TaskRunner subclass — swaps in :class:`CollabLLMRayPPOTrainer`.

verl's stock ``TaskRunner.run`` constructs ``RayPPOTrainer`` directly,
so the only way to use a different trainer class is to override
``run`` in a subclass. The body below is a near-verbatim copy of the
parent's ``run`` (verl/trainer/main_ppo.py) with one line changed —
the trainer instantiation. We accept the duplication because it
isolates all CollabLLM logic to this recipe and avoids any
modification to verl framework files.
"""

from __future__ import annotations

import os
import socket

from verl.trainer.main_ppo import (
    TaskRunner,
    create_rl_dataset,
    create_rl_sampler,
)
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

from .trainer import CollabLLMRayPPOTrainer


class CollabLLMTaskRunner(TaskRunner):
    """Same as TaskRunner but instantiates CollabLLMRayPPOTrainer."""

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"[CollabLLM] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
            is_train=True, max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
            is_train=False, max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # The only line that changes vs. parent: our trainer subclass.
        trainer = CollabLLMRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()
