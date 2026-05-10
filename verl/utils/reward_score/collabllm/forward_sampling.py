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
"""Forward-sampling pool + simulation loop + metric aggregation.

Core of CollabLLM's multi-turn aware reward.

  1. From N rollout responses, build N*B trajectory entries.
  2. Repeat for at most ``window`` turns:
       a. Drop entries that hit the window cap.
       b. Parallel User-Simulator round on every active entry.
          (Per-entry threaded — each entry has different STP context.)
       c. *Batched* Policy round on still-active entries.
          (One ``policy_caller.generate_batch`` call covers everyone.)
  3. Score every trajectory on every metric in parallel.
  4. Aggregate metric scores into r_star (weighted sum) per branch,
     then average branches into per-response MR.

The Policy round is round-level batched. This is what enables the
trainer-injected ``GenFnPolicyCaller`` to use the live actor's vLLM
efficiently — all active conversations across the whole batch are
fanned into a single ``actor_rollout_wg.generate_sequences`` call per
turn. The HTTP fallback uses the same interface and fans out internally
via ThreadPool.
"""

from __future__ import annotations

import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import CollabLLMConfig
from .llm_client import LLMClient
from .metrics import _get_encoding, score_one
from .policy_caller import PolicyCaller
from .prompts import render_user_simulator_prompt, safe_parse_json
from .trajectory import (
    TERMINAL_POLICY_ERROR,
    TERMINAL_SIMULATOR_ERROR,
    TERMINAL_TOKEN_BUDGET,
    TERMINAL_USER_SATISFIED,
    TERMINAL_WINDOW_EXHAUSTED,
    TrajectoryEntry,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Pool initialization
# ----------------------------------------------------------------------
def init_pool(
    rollout_pairs: list[tuple[list[dict[str, str]], str]],
    branches: int,
) -> list[TrajectoryEntry]:
    """Build the initial trajectory pool.

    Each branch gets a *deep copy* of the prefix — branches share the
    same prompt+response initially and would otherwise mutate each other
    once messages start being appended.
    """
    pool: list[TrajectoryEntry] = []
    for origin_id, (prompt_msgs, response_text) in enumerate(rollout_pairs):
        for branch_id in range(branches):
            convo = copy.deepcopy(prompt_msgs)
            convo.append({"role": "assistant", "content": response_text})
            pool.append(
                TrajectoryEntry(
                    origin_id=origin_id,
                    branch_id=branch_id,
                    conversation=convo,
                )
            )
    return pool


# ----------------------------------------------------------------------
# User-simulator step (per-entry, threaded — each has a different STP)
# ----------------------------------------------------------------------
def _simulator_step(
    entry: TrajectoryEntry,
    *,
    single_turn_prompt: str,
    config: CollabLLMConfig,
    sim_client: LLMClient,
) -> None:
    """One user-simulator turn on one entry. Mutates ``entry`` in place."""
    prompt = render_user_simulator_prompt(
        task_desc=config.task_desc,
        single_turn_prompt=single_turn_prompt,
        conversation=entry.conversation,
        terminal_signal=config.terminal_signal,
    )
    try:
        raw = sim_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=config.llm_model,
            temperature=config.user_simulator_temperature,
            max_tokens=config.user_simulator_max_tokens,
            json_mode=True,
            retries=config.api_retries,
            initial_backoff=config.api_initial_backoff,
            tag="user_simulator",
            meta={
                "origin_id": entry.origin_id,
                "branch_id": entry.branch_id,
                "turn": entry.turn_count,
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "User simulator failed (origin=%d, branch=%d): %s",
            entry.origin_id, entry.branch_id, e,
        )
        entry.stop(TERMINAL_SIMULATOR_ERROR)
        return

    parsed = safe_parse_json(raw, default=None)
    if isinstance(parsed, dict) and "response" in parsed:
        user_reply = str(parsed["response"]).strip()
    else:
        user_reply = (raw or "").strip()
        if not user_reply:
            entry.stop(TERMINAL_SIMULATOR_ERROR)
            return

    if config.terminal_signal in user_reply:
        cleaned = user_reply.replace(config.terminal_signal, "").strip()
        if cleaned:
            entry.append("user", cleaned)
        entry.stop(TERMINAL_USER_SATISFIED)
        return

    entry.append("user", user_reply)


def _run_simulator_round(
    active: list[TrajectoryEntry],
    *,
    stp_by_origin: list[str],
    config: CollabLLMConfig,
    sim_client: LLMClient,
) -> None:
    """One simulator round across all active entries, in parallel threads."""
    if not active:
        return

    def step(e: TrajectoryEntry) -> None:
        _simulator_step(
            e,
            single_turn_prompt=stp_by_origin[e.origin_id],
            config=config,
            sim_client=sim_client,
        )

    with ThreadPoolExecutor(max_workers=config.max_simulator_workers) as ex:
        futs = {ex.submit(step, e): e for e in active}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                entry = futs[fut]
                logger.error(
                    "Simulator round crashed (origin=%d branch=%d): %s",
                    entry.origin_id, entry.branch_id, e,
                )
                entry.stop(TERMINAL_SIMULATOR_ERROR)


# ----------------------------------------------------------------------
# Policy round (batched at round level)
# ----------------------------------------------------------------------
def _run_policy_round(
    active: list[TrajectoryEntry],
    *,
    config: CollabLLMConfig,
    policy_caller: PolicyCaller,
) -> None:
    """One policy round across all active entries, *batched* through PolicyCaller.

    All active conversations are submitted in a single
    ``policy_caller.generate_batch`` call. The caller is responsible for
    its own concurrency (HTTP fan-out / vLLM batching).
    """
    if not active:
        return

    convs = [e.conversation for e in active]
    metas = [
        {
            "origin_id": e.origin_id,
            "branch_id": e.branch_id,
            "turn": e.turn_count,
        }
        for e in active
    ]
    replies = policy_caller.generate_batch(convs, meta_batch=metas)

    enc = _get_encoding(config.tiktoken_encoding)
    for entry, reply in zip(active, replies):
        if not reply or not reply.strip():
            entry.stop(TERMINAL_POLICY_ERROR)
            continue
        entry.append("assistant", reply.strip())
        entry.turn_count += 1
        # Token-budget guard. Approximate using assistant + user content.
        total = sum(len(enc.encode(m.get("content", ""))) for m in entry.conversation)
        if total >= config.max_seq_len:
            entry.stop(TERMINAL_TOKEN_BUDGET)


# ----------------------------------------------------------------------
# Whole-pool simulation
# ----------------------------------------------------------------------
def _simulate(
    pool: list[TrajectoryEntry],
    *,
    stp_by_origin: list[str],
    config: CollabLLMConfig,
    sim_client: LLMClient,
    policy_caller: PolicyCaller,
) -> None:
    """Advance the whole pool up to the window cap.

    Window+1 outer iterations is intentional: it lets the user simulator
    have the *last* word after the policy's window-th turn (so the
    trajectory ends with a user message, mimicking real conversation).
    """
    for _ in range(config.forward_sampling_window + 1):
        for e in pool:
            if e.is_active and e.turn_count >= config.forward_sampling_window:
                e.stop(TERMINAL_WINDOW_EXHAUSTED)

        active = [e for e in pool if e.is_active]
        if not active:
            return
        _run_simulator_round(active, stp_by_origin=stp_by_origin,
                             config=config, sim_client=sim_client)

        active = [e for e in pool if e.is_active]
        if not active:
            return
        _run_policy_round(active, config=config, policy_caller=policy_caller)


# ----------------------------------------------------------------------
# Metric scoring (parallel fan-out over entries × metrics)
# ----------------------------------------------------------------------
def _score(
    pool: list[TrajectoryEntry],
    *,
    stp_by_origin: list[str],
    gt_by_origin: list[str],
    config: CollabLLMConfig,
    judge_client: LLMClient,
) -> None:
    """Fill ``entry.scores`` and ``entry.r_star`` for every entry in place."""
    weights = dict(zip(config.metric_names, config.metric_weights))
    tasks = [(e, m) for e in pool for m in config.metric_names]

    with ThreadPoolExecutor(max_workers=config.max_metric_workers) as ex:
        futs = {
            ex.submit(
                score_one,
                m,
                e,
                single_turn_prompt=stp_by_origin[e.origin_id],
                ground_truth=gt_by_origin[e.origin_id],
                config=config,
                judge_client=judge_client,
            ): (e, m)
            for e, m in tasks
        }
        for fut in as_completed(futs):
            e, m = futs[fut]
            try:
                e.scores[m] = float(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.error("metric %s failed (origin=%d branch=%d): %s",
                             m, e.origin_id, e.branch_id, exc)
                if m == "accuracy":
                    e.scores[m] = float(config.accuracy_default)
                elif m == "interactivity":
                    e.scores[m] = float(config.interactivity_default)
                else:
                    e.scores[m] = 0.0

    for e in pool:
        e.r_star = sum(weights[m] * e.scores.get(m, 0.0) for m in config.metric_names)


# ----------------------------------------------------------------------
# Public entry point: rollout pairs -> MR per response
# ----------------------------------------------------------------------
def compute_multiturn_rewards(
    rollout_pairs: list[tuple[list[dict[str, str]], str]],
    *,
    single_turn_prompts: list[str],
    ground_truths: list[str],
    config: CollabLLMConfig,
    sim_client: LLMClient,
    policy_caller: PolicyCaller,
    judge_client: LLMClient,
) -> tuple[list[float], list[dict]]:
    """Run the full pipeline and return MR per rollout pair.

    Args:
        rollout_pairs: N pairs of (prompt history, response text).
        single_turn_prompts: per-pair original full question.
        ground_truths: per-pair reference answer.
        config: tuning knobs.
        sim_client / judge_client: chat clients for User Simulator and
            LLM Judges (typically the same provider/object).
        policy_caller: abstraction over the policy turn — either an
            ``HTTPPolicyCaller`` (test path) or a ``GenFnPolicyCaller``
            wrapping the trainer's live actor (production path).

    Returns:
        (mr_values, debug_info) — N MR floats and per-response diagnostic dicts.
    """
    n = len(rollout_pairs)
    if not (len(single_turn_prompts) == n == len(ground_truths)):
        raise ValueError(
            f"length mismatch: rollout_pairs={n}, "
            f"single_turn_prompts={len(single_turn_prompts)}, "
            f"ground_truths={len(ground_truths)}"
        )

    pool = init_pool(rollout_pairs, branches=config.forward_sampling_branches)

    _simulate(
        pool,
        stp_by_origin=list(single_turn_prompts),
        config=config,
        sim_client=sim_client,
        policy_caller=policy_caller,
    )

    _score(
        pool,
        stp_by_origin=list(single_turn_prompts),
        gt_by_origin=list(ground_truths),
        config=config,
        judge_client=judge_client,
    )

    # Aggregate branches into per-response MR.
    branch_buckets: dict[int, list[TrajectoryEntry]] = {}
    for e in pool:
        branch_buckets.setdefault(e.origin_id, []).append(e)

    mr_values: list[float] = [0.0] * n
    debug_info: list[dict] = []
    for origin_id in range(n):
        siblings = branch_buckets.get(origin_id, [])
        if not siblings:
            mr_values[origin_id] = 0.0
            debug_info.append({"branches": 0})
            continue
        rs = [s.r_star for s in siblings if s.r_star is not None]
        mr_values[origin_id] = sum(rs) / len(rs) if rs else 0.0

        metric_avg: dict[str, float] = {}
        for m in config.metric_names:
            vals = [s.scores.get(m, 0.0) for s in siblings]
            metric_avg[m] = sum(vals) / len(vals) if vals else 0.0
        debug_info.append({
            "branches": len(siblings),
            "metric_avg": metric_avg,
            "terminal_reasons": [s.terminal_reason or "active" for s in siblings],
            "r_stars": [s.r_star for s in siblings],
        })

    return mr_values, debug_info
