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
"""Prompt templates for User Simulator and LLM Judges in CollabLLM."""

from __future__ import annotations

import json
from typing import Any


USER_SIMULATOR_TEMPLATE = """You are role-playing a real human user who is interacting with an AI assistant. \
You are NOT the AI assistant.

# Task type
{task_desc}

# Your hidden goal (the original full question you actually want to solve)
{single_turn_prompt}

# Conversation so far
{conversation_str}

# Behavior guidelines
1. Stay in character as a human user. Never break character.
2. You may have only a partial idea of what you want and may express it imperfectly. \
Real users sometimes make calculation mistakes, forget conditions, or give incomplete \
information. Mimic that behavior occasionally (about 30% of turns).
3. Minimize your own effort. Reply with short, natural utterances. Do not write essays.
4. Be goal-oriented: respond in a way that moves you toward your hidden goal. \
If the assistant asks a clarifying question, answer it briefly.
5. If the assistant has produced a final answer that you believe successfully solves \
your hidden goal, AND no further clarification is needed, output the literal string \
{terminal_signal} in your `response` field to end the conversation. Otherwise continue \
the conversation naturally.
6. Output strictly valid JSON with two fields: `thought` (your private reasoning, \
1-2 sentences) and `response` (what you actually say to the assistant).

# Output format (must be valid JSON, no extra text outside the JSON)
{{"thought": "<your private reasoning>", "response": "<your reply or {terminal_signal}>"}}"""


ACCURACY_JUDGE_TEMPLATE = """You are an evaluator. Decide whether the model's final \
answer correctly solves the target question, given the ground-truth answer.

# Target question
{single_turn_prompt}

# Ground truth
{ground_truth}

# Model's final response (this is the assistant's last message in the conversation)
{completion}

# Scoring criteria
- 1: model's final response contains the correct answer (semantically equivalent to \
ground truth; minor wording differences are acceptable; partial credit is NOT given).
- 0: model's final response is missing, wrong, contradictory, or the conversation \
ended without a final answer.

# Output format (must be valid JSON, no extra text)
{{"thought": "<one short paragraph of reasoning>", "accuracy": <0 or 1>}}"""


INTERACTIVITY_JUDGE_TEMPLATE = """You are an evaluator. Score the AI assistant's \
*interactivity quality* across the following multi-turn conversation.

# Full conversation transcript
{transcript}

# Scoring rubric (return a float in [0, 1])
- 1.0: assistant proactively clarifies ambiguity, asks targeted follow-up questions \
when user input is incomplete, gracefully corrects user mistakes without being \
condescending, and engages the user in collaborative problem solving.
- 0.5: assistant is helpful but mostly passive — answers what is asked but does not \
guide the user.
- 0.0: assistant is verbose, dumps long monologues, ignores user signals, fails to \
clarify, or talks past the user.

Be strict. Most acceptable but uninspiring conversations should score around 0.4-0.6.

# Output format (must be valid JSON, no extra text)
{{"thought": "<one short paragraph of reasoning>", "interactivity": <float in [0,1]>}}"""


def _format_conversation(messages: list[dict[str, str]]) -> str:
    """Format a list of {role, content} messages into a readable transcript."""
    lines = []
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


def render_user_simulator_prompt(
    task_desc: str,
    single_turn_prompt: str,
    conversation: list[dict[str, str]],
    terminal_signal: str,
) -> str:
    """Build the user-simulator prompt by filling the template."""
    return USER_SIMULATOR_TEMPLATE.format(
        task_desc=task_desc,
        single_turn_prompt=single_turn_prompt,
        conversation_str=_format_conversation(conversation),
        terminal_signal=terminal_signal,
    )


def render_accuracy_judge_prompt(
    single_turn_prompt: str,
    ground_truth: str,
    completion: str,
) -> str:
    """Build the accuracy-judge prompt."""
    return ACCURACY_JUDGE_TEMPLATE.format(
        single_turn_prompt=single_turn_prompt,
        ground_truth=ground_truth,
        completion=completion,
    )


def render_interactivity_judge_prompt(conversation: list[dict[str, str]]) -> str:
    """Build the interactivity-judge prompt."""
    return INTERACTIVITY_JUDGE_TEMPLATE.format(
        transcript=_format_conversation(conversation),
    )


def safe_parse_json(text: str, default: Any) -> Any:
    """Best-effort JSON parsing with markdown-fence stripping. Returns ``default`` on failure."""
    if not isinstance(text, str):
        return default
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return default
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return default
