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
"""Policy caller — abstraction over "how does forward sampling get its
policy turn?".

Two implementations:

  HTTPPolicyCaller
    Calls an OpenAI-compatible vLLM server. Used for *standalone testing*
    (e.g. tests/collabllm/test_pipeline.py) where the trainer isn't
    running and we don't have access to ``actor_rollout_wg``. Internally
    fans out to a ThreadPool so a batch of N conversations turns into N
    concurrent HTTP requests — vLLM batches them server-side.

  GenFnPolicyCaller
    Wraps a callable injected by the verl trainer (recipes/collabllm).
    The callable proxies to ``actor_rollout_wg.generate_sequences`` so
    forward sampling uses *the live actor's vLLM with the current
    weights* — no checkpoint drift. The callable already accepts a
    batch, so we just delegate.

The two implementations share the same ``generate_batch`` signature:
``list[messages] -> list[reply]``. forward_sampling.py talks to the
abstraction and doesn't care which path is wired up.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class PolicyCaller(ABC):
    """Take N conversations, return N assistant replies."""

    @abstractmethod
    def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        meta_batch: list[dict] | None = None,
    ) -> list[str]:
        """Run the policy on every conversation in the batch.

        Args:
            messages_batch: list of OpenAI-format chat histories.
            meta_batch: optional per-item metadata (origin/branch/turn)
                for tracing. Length must match ``messages_batch`` if given.

        Returns:
            One reply string per input. Empty string indicates failure
            for that item — caller is expected to mark the entry as
            terminal (TERMINAL_POLICY_ERROR).
        """
        raise NotImplementedError


class HTTPPolicyCaller(PolicyCaller):
    """Standalone path: each conversation → one HTTP request to a vLLM
    server. Concurrent fan-out via ThreadPool.

    This is what the test pipeline uses (no trainer needed) and what
    the production reward manager falls back to if no ``generation_fn``
    is injected. In production with the recipe entry point, this is
    NOT the active path — see GenFnPolicyCaller.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float | None,
        retries: int = 3,
        initial_backoff: float = 1.0,
        max_workers: int = 64,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.retries = retries
        self.initial_backoff = initial_backoff
        self.max_workers = max_workers

    def _one(self, messages: list[dict], meta: dict | None) -> str:
        try:
            return self.client.chat(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                json_mode=False,
                retries=self.retries,
                initial_backoff=self.initial_backoff,
                tag="policy",
                meta=meta,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("HTTPPolicyCaller failed (meta=%s): %s", meta, e)
            return ""

    def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        meta_batch: list[dict] | None = None,
    ) -> list[str]:
        n = len(messages_batch)
        metas: list[dict | None] = list(meta_batch) if meta_batch else [None] * n
        results: list[str] = [""] * n
        if n == 0:
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self._one, messages_batch[i], metas[i]): i for i in range(n)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.error("HTTPPolicyCaller crashed (i=%d): %s", i, e)
                    results[i] = ""
        return results


class GenFnPolicyCaller(PolicyCaller):
    """Production path: forward sampling uses the *live* actor.

    Holds a callable injected by the recipe trainer. The callable's
    contract: take a batch of OpenAI-format conversations, return a list
    of assistant replies (strings). It is constructed in the trainer
    where ``actor_rollout_wg`` is in scope and is responsible for:

      1. Applying the chat template + tokenizing each conversation
      2. Building a DataProto and calling actor_rollout_wg.generate_sequences()
      3. Decoding the output token IDs back to strings

    The trainer-side wrapper batches all entries into a single
    generate_sequences call — that's the whole point of having this
    path: maximum vLLM throughput and zero checkpoint drift.

    Tracing: if a ``trace_writer`` is provided, each (input, output)
    pair is logged after the batch returns. Useful for the recipe to
    keep parity with the HTTP path's trace JSONL.
    """

    def __init__(
        self,
        gen_fn: Callable[[list[list[dict[str, str]]]], list[str]],
        *,
        trace_writer: Callable[[list[dict], str, dict | None], None] | None = None,
    ):
        self.gen_fn = gen_fn
        self._trace_writer = trace_writer

    def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        meta_batch: list[dict] | None = None,
    ) -> list[str]:
        n = len(messages_batch)
        if n == 0:
            return []
        try:
            replies = self.gen_fn(messages_batch)
        except Exception as e:  # noqa: BLE001
            # Trainer-side failure: likely the actor worker errored. Mark
            # everyone as failed; caller will terminate trajectories.
            logger.error("GenFnPolicyCaller failed for whole batch (n=%d): %s", n, e)
            return [""] * n

        if not isinstance(replies, list) or len(replies) != n:
            logger.error(
                "GenFnPolicyCaller: gen_fn returned %d replies for %d requests",
                len(replies) if isinstance(replies, list) else -1, n,
            )
            return [""] * n

        # Optional tracing pass — keeps the JSONL parity with HTTP path.
        if self._trace_writer is not None:
            metas: list[dict | None] = list(meta_batch) if meta_batch else [None] * n
            for messages, reply, meta in zip(messages_batch, replies, metas):
                try:
                    self._trace_writer(messages, reply or "", meta)
                except Exception as e:  # noqa: BLE001
                    logger.warning("trace writer raised: %s", e)

        return [r if isinstance(r, str) else "" for r in replies]
