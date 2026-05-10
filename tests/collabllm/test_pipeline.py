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
"""End-to-end smoke test for the CollabLLM reward pipeline.

This is the *minimum* required to convince yourself the pipeline is
sound before launching verl. It runs the full forward sampling +
multi-metric scoring on a 2-prompt fixture, with every API call logged
to a JSONL trace and a human-readable Markdown report.

Prereqs:
    export DEEPSEEK_API_KEY=...
    bash examples/grpo_trainer/start_reward_vllm.sh   # in another shell

Run:
    python tests/collabllm/test_pipeline.py

Outputs (under /data/nas_tmp/ljd_tmp/collabllm/logs/):
    pipeline_test_<ts>.trace.jsonl   — every LLM request/response
    pipeline_test_<ts>.report.md     — human-readable conversation flow + scores
    pipeline_test_<ts>.summary.json  — per-response MR + per-branch r_star
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make the local verl tree importable without installing it.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verl.utils.reward_score.collabllm.config import CollabLLMConfig  # noqa: E402
from verl.utils.reward_score.collabllm.forward_sampling import (  # noqa: E402
    compute_multiturn_rewards,
)
from verl.utils.reward_score.collabllm.llm_client import (  # noqa: E402
    LLMClient,
    resolve_api_key,
)
from verl.utils.reward_score.collabllm.policy_caller import (  # noqa: E402
    HTTPPolicyCaller,
)


# ----------------------------------------------------------------------
# Fixture: 2 (prompt, response) pairs that mimic real RL rollout output.
# Response #0 is "good" (clarifying/Socratic style); Response #1 is "bad"
# (wrong answer, no clarification). They give the judges something to
# discriminate between, so you can sanity-check whether scoring works.
# ----------------------------------------------------------------------
FIXTURES: list[dict[str, Any]] = [
    {
        "single_turn_prompt": "已知 |a-3| + |b+2| = 0，求 a+b 的值。",
        "ground_truth": "a+b = 1",
        "prompt_messages": [
            {"role": "user", "content": "绝对值的题：|a-3| + |b+2| = 0，求a+b"}
        ],
        "response_text": (
            "因为绝对值非负，两个非负数加起来等于0，必须各自都是0。\n"
            "所以 |a-3|=0 → a=3，|b+2|=0 → b=-2。\n"
            "你算到这一步了吗？最后 a+b 等于多少？"
        ),
    },
    {
        "single_turn_prompt": "已知 |a-3| + |b+2| = 0，求 a+b 的值。",
        "ground_truth": "a+b = 1",
        "prompt_messages": [
            {"role": "user", "content": "绝对值的题：|a-3| + |b+2| = 0，求a+b"}
        ],
        "response_text": "答案是 a+b = 5。",  # 故意错的
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log_dir", default="/data/nas_tmp/ljd_tmp/collabllm/logs",
                   help="where to write the trace JSONL and the Markdown report")
    p.add_argument("--branches", type=int, default=2,
                   help="forward sampling branches per response (small for testing)")
    p.add_argument("--window", type=int, default=2,
                   help="forward sampling turns window")
    p.add_argument("--llm_api_base", default=os.environ.get("LLM_API_BASE", "https://api.deepseek.com"))
    p.add_argument("--llm_api_key_env", default="DEEPSEEK_API_KEY")
    p.add_argument("--llm_model", default=os.environ.get("LLM_MODEL", "deepseek-v4-pro"))
    p.add_argument("--policy_api_base", default=os.environ.get("POLICY_API_BASE", "http://127.0.0.1:8000/v1"))
    p.add_argument("--policy_model", default=os.environ.get("POLICY_MODEL", "collabllm-policy"))
    p.add_argument("--task_desc", default="math problem solving with collaborative tutoring")
    return p.parse_args()


def build_config(args: argparse.Namespace, trace_path: str) -> CollabLLMConfig:
    return CollabLLMConfig(
        forward_sampling_window=args.window,
        forward_sampling_branches=args.branches,
        terminal_signal="[TERMINATE]",
        task_desc=args.task_desc,
        metric_names=("accuracy", "token_amount", "interactivity"),
        metric_weights=(1.0, -0.5, 1.0),
        llm_api_base=args.llm_api_base,
        llm_api_key_env=args.llm_api_key_env,
        llm_model=args.llm_model,
        policy_api_base=args.policy_api_base,
        policy_api_key="EMPTY",
        policy_model=args.policy_model,
        trace_path=trace_path,
        max_metric_workers=8,
        max_simulator_workers=8,
        max_policy_workers=8,
        api_retries=3,
    )


def preflight_checks(args: argparse.Namespace) -> None:
    """Fail fast with actionable errors before burning API quota."""
    key = os.environ.get(args.llm_api_key_env, "").strip()
    if not key:
        raise SystemExit(
            f"ERROR: env var {args.llm_api_key_env} is empty. Export it before running."
        )
    # Probe the policy endpoint — vLLM should answer /v1/models
    import urllib.request
    try:
        with urllib.request.urlopen(
            args.policy_api_base.rstrip("/") + "/models", timeout=5
        ) as resp:
            assert resp.status == 200
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"ERROR: policy endpoint {args.policy_api_base} not reachable: {e}\n"
            "Start it: bash examples/grpo_trainer/start_reward_vllm.sh"
        ) from e


def run(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = log_dir / f"pipeline_test_{ts}.trace.jsonl"
    report_path = log_dir / f"pipeline_test_{ts}.report.md"
    summary_path = log_dir / f"pipeline_test_{ts}.summary.json"

    print(f"[test] trace   → {trace_path}")
    print(f"[test] report  → {report_path}")
    print(f"[test] summary → {summary_path}")

    cfg = build_config(args, trace_path=str(trace_path))

    api_key = resolve_api_key(cfg.llm_api_key_env, fallback="EMPTY")
    sim_client = LLMClient.get_shared(
        base_url=cfg.llm_api_base, api_key=api_key,
        request_timeout=cfg.api_request_timeout, trace_path=cfg.trace_path,
    )
    policy_client = LLMClient.get_shared(
        base_url=cfg.policy_api_base, api_key=cfg.policy_api_key,
        request_timeout=cfg.api_request_timeout, trace_path=cfg.trace_path,
    )

    # Standalone test uses the HTTP-based PolicyCaller; production
    # training uses GenFnPolicyCaller via the recipe trainer.
    policy_caller = HTTPPolicyCaller(
        client=policy_client,
        model=cfg.policy_model,
        temperature=cfg.policy_temperature,
        max_tokens=cfg.policy_max_tokens,
        top_p=cfg.policy_top_p,
        retries=cfg.api_retries,
        initial_backoff=cfg.api_initial_backoff,
        max_workers=cfg.max_policy_workers,
    )

    rollout_pairs = [(f["prompt_messages"], f["response_text"]) for f in FIXTURES]
    single_turn_prompts = [f["single_turn_prompt"] for f in FIXTURES]
    ground_truths = [f["ground_truth"] for f in FIXTURES]

    print(f"[test] running pipeline on {len(FIXTURES)} fixtures, "
          f"branches={cfg.forward_sampling_branches}, window={cfg.forward_sampling_window}")

    mr_values, debug_info = compute_multiturn_rewards(
        rollout_pairs=rollout_pairs,
        single_turn_prompts=single_turn_prompts,
        ground_truths=ground_truths,
        config=cfg,
        sim_client=sim_client,
        policy_caller=policy_caller,
        judge_client=sim_client,  # same provider; same client
    )

    # ---------- summary JSON ----------
    summary = {
        "config": {
            "model": cfg.llm_model,
            "policy_model": cfg.policy_model,
            "branches": cfg.forward_sampling_branches,
            "window": cfg.forward_sampling_window,
            "metric_names": list(cfg.metric_names),
            "metric_weights": list(cfg.metric_weights),
        },
        "results": [
            {
                "origin_id": i,
                "single_turn_prompt": single_turn_prompts[i],
                "ground_truth": ground_truths[i],
                "response": rollout_pairs[i][1],
                "MR": mr_values[i],
                "debug": debug_info[i],
            }
            for i in range(len(FIXTURES))
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Markdown report ----------
    md = build_report(
        cfg=cfg, fixtures=FIXTURES, mr_values=mr_values,
        debug_info=debug_info, trace_path=str(trace_path),
    )
    report_path.write_text(md, encoding="utf-8")

    # ---------- console summary ----------
    print()
    print("=" * 72)
    print(f"{'origin':<8} {'MR':>10}   {'accuracy':>10} {'tokens(k)':>10} {'interact':>10}")
    print("-" * 72)
    for i, info in enumerate(debug_info):
        avg = info.get("metric_avg", {})
        print(
            f"{i:<8} {mr_values[i]:>10.4f}   "
            f"{avg.get('accuracy', 0):>10.3f} {avg.get('token_amount', 0):>10.3f} "
            f"{avg.get('interactivity', 0):>10.3f}"
        )
    print("=" * 72)
    print(f"\nWrote {trace_path.name}, {report_path.name}, {summary_path.name}")


# ----------------------------------------------------------------------
# Markdown report — reads the trace JSONL we just wrote and stitches
# together a per-response narrative (turn-by-turn conversation, judge
# verdicts with their `thought` fields, weighted r_star arithmetic).
# ----------------------------------------------------------------------
def build_report(
    *,
    cfg: CollabLLMConfig,
    fixtures: list[dict[str, Any]],
    mr_values: list[float],
    debug_info: list[dict],
    trace_path: str,
) -> str:
    # Re-read the trace and bucket events by (origin_id, branch_id).
    events: dict[tuple[int, int], list[dict]] = {}
    judges: dict[tuple[int, int], list[dict]] = {}
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = rec.get("meta", {})
            oid = meta.get("origin_id")
            bid = meta.get("branch_id")
            if oid is None or bid is None:
                continue
            tag = rec.get("tag", "")
            key = (oid, bid)
            if tag in ("user_simulator", "policy"):
                events.setdefault(key, []).append(rec)
            elif tag in ("accuracy_judge", "interactivity_judge"):
                judges.setdefault(key, []).append(rec)

    lines: list[str] = []
    lines.append(f"# CollabLLM pipeline test report")
    lines.append("")
    lines.append(f"- model: `{cfg.llm_model}`  /  policy: `{cfg.policy_model}`")
    lines.append(f"- branches/response: {cfg.forward_sampling_branches}")
    lines.append(f"- window (turns): {cfg.forward_sampling_window}")
    lines.append(f"- metrics: {list(cfg.metric_names)}")
    lines.append(f"- weights: {list(cfg.metric_weights)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| origin | MR | accuracy | token_amount(k) | interactivity |")
    lines.append("|---|---|---|---|---|")
    for i, info in enumerate(debug_info):
        avg = info.get("metric_avg", {})
        lines.append(
            f"| {i} | **{mr_values[i]:.4f}** | "
            f"{avg.get('accuracy', 0):.3f} | "
            f"{avg.get('token_amount', 0):.3f} | "
            f"{avg.get('interactivity', 0):.3f} |"
        )
    lines.append("")

    weights = dict(zip(cfg.metric_names, cfg.metric_weights))

    for i, fix in enumerate(fixtures):
        lines.append("---")
        lines.append(f"## Response #{i}")
        lines.append("")
        lines.append(f"**Original user prompt**:")
        for msg in fix["prompt_messages"]:
            lines.append(f"> [{msg['role'].upper()}] {msg['content']}")
        lines.append("")
        lines.append(f"**Response being scored (M_j)**:")
        lines.append("```")
        lines.append(fix["response_text"])
        lines.append("```")
        lines.append("")
        lines.append(f"**Hidden goal (single_turn_prompt)**: {fix['single_turn_prompt']}")
        lines.append(f"**Ground truth**: `{fix['ground_truth']}`")
        lines.append(f"**MR (mean of branch r_star)**: **{mr_values[i]:.4f}**")
        lines.append("")
        info = debug_info[i]
        for b in range(cfg.forward_sampling_branches):
            key = (i, b)
            lines.append(f"### Branch {b}")
            lines.append("")
            lines.append(
                f"_terminal_: `{info['terminal_reasons'][b] if b < len(info['terminal_reasons']) else '?'}`"
            )
            lines.append("")
            lines.append("**Forward sampling turns**:")
            lines.append("")
            for ev in events.get(key, []):
                role = "USER (sim)" if ev["tag"] == "user_simulator" else "ASSISTANT (policy)"
                turn = ev.get("meta", {}).get("turn", "?")
                resp = (ev.get("response") or "").strip()
                if ev["tag"] == "user_simulator":
                    parsed = _safe_json(resp)
                    if isinstance(parsed, dict) and "response" in parsed:
                        resp = str(parsed["response"]).strip()
                lines.append(f"- **turn {turn} / {role}**:")
                lines.append("")
                lines.append("    ```")
                for ln in resp.splitlines() or [""]:
                    lines.append(f"    {ln}")
                lines.append("    ```")
            lines.append("")
            lines.append("**Metric judgments**:")
            lines.append("")
            for jd in judges.get(key, []):
                tag = jd["tag"]
                resp = (jd.get("response") or "").strip()
                parsed = _safe_json(resp)
                thought = ""
                value = None
                if isinstance(parsed, dict):
                    thought = str(parsed.get("thought", "")).strip()
                    if tag == "accuracy_judge":
                        value = parsed.get("accuracy")
                    elif tag == "interactivity_judge":
                        value = parsed.get("interactivity")
                lines.append(f"- **{tag}** → value: `{value}`")
                if thought:
                    lines.append(f"  - judge thought: _{thought}_")
            # Pull out per-metric scores (incl. token_amount which has no
            # judge call but still has a numeric score in debug_info).
            r_stars = info.get("r_stars", [])
            r_star = r_stars[b] if b < len(r_stars) else None
            metric_avg = info.get("metric_avg", {})  # branch-level not stored, only avg
            lines.append("")
            if r_star is not None:
                weighted_terms = []
                for m in cfg.metric_names:
                    weighted_terms.append(f"{weights[m]:+.2f}*{m}")
                lines.append(f"- **r_star** = {r_star:.4f}  (formula: {' '.join(weighted_terms)})")
            lines.append("")

    lines.append("---")
    lines.append(f"_Full request/response trace: `{Path(trace_path).name}`_")
    return "\n".join(lines) + "\n"


def _safe_json(text: str):
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return None
    try:
        return json.loads(s[a : b + 1])
    except json.JSONDecodeError:
        return None


def main() -> None:
    args = parse_args()
    preflight_checks(args)
    run(args)


if __name__ == "__main__":
    main()
