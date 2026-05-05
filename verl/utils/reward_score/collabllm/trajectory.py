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
"""Trajectory entry — the unit of state in the forward-sampling pool.

One ``TrajectoryEntry`` represents one (response, branch) tuple as it
evolves turn by turn: the conversation grows, ``turn_count`` advances,
and ``is_active`` flips off when a terminal condition is reached.

Implementation note: ``conversation`` is mutated in-place during
simulation, so each branch must be initialized with a *deep copy* of
the prefix (handled by :func:`init_pool`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Reasons a trajectory may stop simulating before the window is exhausted.
TERMINAL_USER_SATISFIED = "user_satisfied"
TERMINAL_WINDOW_EXHAUSTED = "window_exhausted"
TERMINAL_POLICY_ERROR = "policy_error"
TERMINAL_TOKEN_BUDGET = "token_budget"
TERMINAL_SIMULATOR_ERROR = "simulator_error"


@dataclass
class TrajectoryEntry:
    """One (response, branch) trajectory.

    Attributes:
        origin_id: index of the source response in the rollout batch
            (range: 0..num_responses-1). Used to aggregate branches back
            to their parent response.
        branch_id: index of this branch among its siblings (range: 0..B-1).
        conversation: list of ``{"role": str, "content": str}`` messages,
            in OpenAI chat format. Initially the rollout prefix, grows
            during simulation.
        turn_count: number of *forward* turns completed (a turn = one user
            simulator call + one policy call). Capped at config.window.
        is_active: whether this trajectory should keep simulating. Flipped
            to False on terminal events.
        terminal_reason: one of the TERMINAL_* constants above, or "" while
            still active.
        scores: per-metric raw scores, populated by the metric scorers
            after simulation finishes (e.g., {"accuracy": 1.0, ...}).
        r_star: the final weighted reward for this branch (filled by the
            metric aggregator).
    """

    origin_id: int
    branch_id: int
    conversation: list[dict[str, str]] = field(default_factory=list)
    turn_count: int = 0
    is_active: bool = True
    terminal_reason: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    r_star: float | None = None

    def stop(self, reason: str) -> None:
        """Mark this entry as terminal with a recorded reason."""
        self.is_active = False
        if not self.terminal_reason:
            self.terminal_reason = reason

    def append(self, role: str, content: str) -> None:
        """Append a turn to the conversation in OpenAI chat format."""
        self.conversation.append({"role": role, "content": content})

    def last_assistant(self) -> str:
        """Return the most recent assistant message text, or '' if none."""
        for msg in reversed(self.conversation):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""
