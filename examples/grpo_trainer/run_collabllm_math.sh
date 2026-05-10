#!/usr/bin/env bash
# GRPO + CollabLLM multi-turn aware reward, on 2x RTX 4090 (48GB).
#
# Routes through the CollabLLM recipe entry point so forward sampling
# uses the *live actor's vLLM* (zero checkpoint drift) — no side-car
# vLLM server is required for production training.
#
# Pipeline (one full step):
#   rollout (vLLM, n=8) → recompute log_prob (FSDP) → log_ref (FSDP)
#       → reward [forward sampling via live actor + DeepSeek judges]
#       → GRPO advantage (group z-score, in verl) → ppo_epochs updates
#       → sync weights to vLLM → next step
#
# Hardware layout (GPUs 0 and 1, both 48GB):
#   - actor FSDP shard:       ~6 GB / GPU
#   - actor rollout vLLM:     ~14 GB / GPU  (gpu_memory_utilization=0.65)
#   - ref policy (offloaded): ~1 GB / GPU
#   - headroom:               ~25 GB / GPU
#
# Prereqs:
#   1. SFT-merged checkpoint at SFT_MERGED_PATH.
#   2. DEEPSEEK_API_KEY exported.
#   3. wandb logged in.
#
# NOT required for production:
#   - The side-car reward vLLM (start_reward_vllm.sh). That script is
#     now only used by tests/collabllm/test_pipeline.py for offline
#     pipeline validation.
#
# Launch:
#   bash examples/grpo_trainer/run_collabllm_math.sh

set -xeuo pipefail

# ---------- paths ----------
SFT_MERGED_PATH=${SFT_MERGED_PATH:-/data/nas_tmp/ljd_tmp/collabllm/merged/sft_lora_v1_real_merged}
TRAIN_PARQUET=${TRAIN_PARQUET:-/data/nas_tmp/ljd_tmp/collabllm/data/rl_math/rl_math_train.parquet}
CKPT_DIR=${CKPT_DIR:-/data/nas_tmp/ljd_tmp/collabllm/checkpoints}
LOG_DIR=${LOG_DIR:-/data/nas_tmp/ljd_tmp/collabllm/logs}
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

# ---------- experiment identity ----------
PROJECT_NAME=${PROJECT_NAME:-collabllm}
EXP_NAME=${EXP_NAME:-collabllm_grpo_math_$(date +%Y%m%d_%H%M%S)}
WANDB_ENTITY=${WANDB_ENTITY:-lujiadong-nus}

# ---------- LLM judge / user simulator (DeepSeek) ----------
LLM_API_BASE=${LLM_API_BASE:-https://api.deepseek.com}
LLM_API_KEY_ENV=${LLM_API_KEY_ENV:-DEEPSEEK_API_KEY}
LLM_MODEL=${LLM_MODEL:-deepseek-v4-flash}

if [ -z "${!LLM_API_KEY_ENV:-}" ]; then
    echo "ERROR: ${LLM_API_KEY_ENV} env var is empty. Export it before launching."
    exit 1
fi

# ---------- main launch ----------
WANDB_ENTITY=${WANDB_ENTITY} \
CUDA_VISIBLE_DEVICES=0,1 \
python -m recipes.collabllm.main_collabllm_ppo \
    \
    `# --- algorithm ---` \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    `# --- data ---` \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files=null \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.truncation=right \
    \
    `# --- actor (FSDP) ---` \
    actor_rollout_ref.model.path="${SFT_MERGED_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    `# --- rollout (vLLM, n=8 GRPO group; same vLLM also serves forward sampling) ---` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    `# --- reference policy (frozen, offloaded) ---` \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    \
    `# --- reward manager (CollabLLM, inline dispatch) ---` \
    reward.reward_manager.source=register \
    reward.reward_manager.name=collabllm \
    +reward.reward_kwargs.forward_sampling_window=2 \
    +reward.reward_kwargs.forward_sampling_branches=3 \
    +reward.reward_kwargs.metric_names='[accuracy,token_amount,interactivity]' \
    +reward.reward_kwargs.metric_weights='[1.0,-0.5,1.0]' \
    +reward.reward_kwargs.terminal_signal='[TERMINATE]' \
    +reward.reward_kwargs.task_desc='math problem solving with collaborative tutoring' \
    +reward.reward_kwargs.max_seq_len=4096 \
    +reward.reward_kwargs.token_amount_clip_k=4.0 \
    +reward.reward_kwargs.llm_api_base="${LLM_API_BASE}" \
    +reward.reward_kwargs.llm_api_key_env="${LLM_API_KEY_ENV}" \
    +reward.reward_kwargs.llm_model="${LLM_MODEL}" \
    +reward.reward_kwargs.user_simulator_temperature=0.8 \
    +reward.reward_kwargs.judge_temperature=0.0 \
    +reward.reward_kwargs.max_metric_workers=64 \
    +reward.reward_kwargs.max_simulator_workers=64 \
    +reward.reward_kwargs.api_retries=3 \
    +reward.reward_kwargs.api_initial_backoff=1.0 \
    +reward.reward_kwargs.api_request_timeout=60.0 \
    \
    `# --- trainer ---` \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.logger='["console","wandb"]' \
    trainer.default_local_dir="${CKPT_DIR}/${PROJECT_NAME}/${EXP_NAME}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.total_epochs=4 \
    +trainer.max_ckpt_to_keep=3 \
    trainer.resume_mode=auto \
    \
    2>&1 | tee "${LOG_DIR}/${EXP_NAME}.log"
