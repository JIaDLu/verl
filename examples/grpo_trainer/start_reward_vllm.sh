#!/usr/bin/env bash
# Start a small vLLM HTTP server that serves the SFT-merged checkpoint as
# the "policy" used inside CollabLLM forward sampling.
#
# Why a separate server?
#   The reward function needs to call the policy mid-step to simulate
#   future turns. The actor's own vLLM is busy/asleep during the reward
#   phase and is not exposed as an HTTP endpoint to outside processes,
#   so we run a lightweight, frozen copy here. Trade-off documented in
#   run_collabllm_math.sh.
#
# Memory budget:
#   1.5B bf16 ≈ 3GB params + ~5-8GB KV cache for short forward windows.
#   We squeeze it into ~25% of one 4090 so the actor can still use the
#   rest. Tune --gpu-memory-utilization if it OOMs.
#
# Usage:
#   bash start_reward_vllm.sh
#
# Stop it later:
#   pkill -f "served-model-name collabllm-policy"

set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-/data/nas_tmp/ljd_tmp/collabllm/merged/sft_lora_v1_real_merged}
SERVED_NAME=${SERVED_NAME:-collabllm-policy}
PORT=${PORT:-8000}
GPU_ID=${GPU_ID:-1}                    # share GPU 1 with training
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.25}     # reserve ~12 GB on a 48 GB card
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
LOG_DIR=${LOG_DIR:-/data/nas_tmp/ljd_tmp/collabllm/logs}
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${GPU_ID} \
nohup python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --dtype bfloat16 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enforce-eager \
    > "${LOG_DIR}/reward_vllm_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

PID=$!
echo "Started reward vLLM (pid=${PID}) on GPU ${GPU_ID}, port ${PORT}"
echo "Waiting for /v1/models to become ready..."

for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "Reward vLLM ready."
        exit 0
    fi
    sleep 2
done

echo "Reward vLLM failed to come up within 120s. Check ${LOG_DIR}/reward_vllm_*.log"
exit 1
