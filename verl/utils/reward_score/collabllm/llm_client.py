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
"""Unified OpenAI-compatible chat client for CollabLLM.

A single thin wrapper around the ``openai`` SDK so all three call sites
(User Simulator, LLM Judges, vLLM Policy) share retry / backoff /
timeout / JSON-mode behavior.

Why not aiohttp / requests directly?
  The ``openai`` library handles streaming, JSON mode, and base-URL
  swapping uniformly. Using it for both the API provider (gpt-5.2) and
  the local vLLM server keeps one code path — vLLM exposes an
  OpenAI-compatible endpoint out of the box.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient:
    """Thread-safe blocking client for OpenAI-compatible chat completions.

    Reuses a single ``openai.OpenAI`` instance per (base_url, api_key);
    the SDK is documented as thread-safe so we serialize neither the
    client nor the underlying httpx pool.

    Retries on any exception with exponential backoff. After exhausting
    retries the caller decides what to do — usually return a default
    score and emit a warning rather than fail the whole training step.
    """

    _shared: dict[tuple[str, str], "LLMClient"] = {}
    _shared_lock = threading.Lock()

    def __init__(self, base_url: str, api_key: str, request_timeout: float = 60.0):
        # Lazy import — verl users who don't enable CollabLLM shouldn't
        # need the openai package.
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required for CollabLLM. Install with: pip install openai>=1.0"
            ) from e
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)
        self._base_url = base_url

    @classmethod
    def get_shared(cls, base_url: str, api_key: str, request_timeout: float = 60.0) -> "LLMClient":
        """Return a process-wide singleton client per (base_url, api_key)."""
        key = (base_url, api_key)
        with cls._shared_lock:
            client = cls._shared.get(key)
            if client is None:
                client = cls(base_url=base_url, api_key=api_key, request_timeout=request_timeout)
                cls._shared[key] = client
        return client

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        top_p: float | None = None,
        json_mode: bool = False,
        retries: int = 3,
        initial_backoff: float = 1.0,
    ) -> str:
        """Run one chat completion and return the assistant text.

        Args:
            messages: list of {role, content} dicts in OpenAI format.
            model: model name (provider-specific).
            temperature, max_tokens, top_p: sampling knobs.
            json_mode: if True, request response_format=json_object.
                Some providers/models reject this; on TypeError we retry
                without it once.
            retries: number of total attempts including the first.
            initial_backoff: seconds to wait after the first failure;
                doubles each subsequent failure.

        Returns:
            The assistant reply as a string. On total failure raises
            the last exception — caller decides whether to swallow.
        """
        last_exc: Exception | None = None
        backoff = initial_backoff
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                return content
            except TypeError as e:
                # Some providers reject response_format; drop it once and retry.
                if json_mode and "response_format" in kwargs:
                    logger.warning(
                        "LLM API rejected response_format; retrying without JSON mode (%s)", e
                    )
                    kwargs.pop("response_format", None)
                    json_mode = False
                    continue
                last_exc = e
            except Exception as e:  # noqa: BLE001 — we genuinely catch any API error
                last_exc = e
                logger.warning(
                    "LLM call failed (attempt %d/%d, base=%s): %s",
                    attempt + 1,
                    retries,
                    self._base_url,
                    e,
                )
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2

        assert last_exc is not None
        raise last_exc


def resolve_api_key(env_var: str, fallback: str = "EMPTY") -> str:
    """Read an API key from the environment with a sentinel fallback.

    The fallback is convenient for local vLLM (which accepts any string)
    but unlikely to work for real providers — in that case we'd rather
    fail loudly than silently send empty credentials, so we log a warning.
    """
    key = os.environ.get(env_var, "").strip()
    if not key:
        if fallback == "EMPTY":
            logger.warning("Env var %s not set; using sentinel 'EMPTY' (OK for local vLLM only)", env_var)
        return fallback
    return key
