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
"""CollabLLM multi-turn aware reward — public API.

The reward manager (``verl.experimental.reward_loop.reward_manager.collabllm``)
glues these pieces together; you typically don't import from here directly
unless extending or testing the pipeline.
"""

from .config import CollabLLMConfig
from .forward_sampling import compute_multiturn_rewards, init_pool
from .llm_client import LLMClient, resolve_api_key
from .trajectory import TrajectoryEntry

__all__ = [
    "CollabLLMConfig",
    "LLMClient",
    "TrajectoryEntry",
    "compute_multiturn_rewards",
    "init_pool",
    "resolve_api_key",
]
