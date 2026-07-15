#!/bin/bash
# Per-rank worker for the SPEED-Bench MTP acceptance-rate benchmark.
#
# Launched once per GPU by scripts/speed_bench/run_speed_bench.slurm (via srun,
# inside the mcore container). Every rank starts a Megatron dynamic inference
# server process (they form one distributed engine); rank 0 additionally waits
# for the server to be ready, runs the specdec_bench acceptance-rate client
# against it, then tears the server down.
#
# All configuration comes from environment variables (exported by the sbatch
# script). See scripts/speed_bench/README.md for the full list.
set -uo pipefail

# --- Distributed / per-rank setup -------------------------------------------
export RANK=${RANK:-${SLURM_PROCID:-0}}
LOCAL_RANK=${LOCAL_RANK:-${SLURM_LOCALID:-0}}
export WORLD_SIZE=${WORLD_SIZE:-${SLURM_NTASKS:-1}}
# Rendezvous: default to the first node in the allocation (works for 1 node and
# multi-node). Only set if the launcher didn't already provide them.
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" 2>/dev/null | head -n1)}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-6000}
# Per-rank caches so ranks don't collide writing JIT artifacts.
export TRITON_CACHE_DIR=/tmp/triton_cache_${RANK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_${RANK}
export TORCH_HOME=/tmp/torch_home_${RANK}

# Some container builds export a stale /usr/local/cuda ptxas path. Point
# Triton at the compiler bundled with the installed Triton package so every
# rank can compile cache misses after startup.
TRITON_BUNDLED_PTXAS=/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas
if [ ! -x "${TRITON_BUNDLED_PTXAS}" ]; then
    echo "ERROR: Triton ptxas not found at ${TRITON_BUNDLED_PTXAS}" >&2
    exit 2
fi
export TRITON_PTXAS_PATH="${TRITON_BUNDLED_PTXAS}"

# --- Model / server configuration (env-overridable) -------------------------
CKPT=${CKPT:?set CKPT to your Megatron MTP checkpoint directory}
TOKENIZER=${TOKENIZER:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}
SPEED_BENCH_DATA=${SPEED_BENCH_DATA:?set SPEED_BENCH_DATA to the SPEED-Bench dataset path}
# Previous Nemotron-3-Super inference defaults:
# TP=${TP:-2}
# EP=${EP:-8}
# NanoV3 3B inference uses the single-node TP=1 / EP=8 expert-sharded layout.
TP=${TP:-1}
EP=${EP:-8}
CKPT_STEP=${CKPT_STEP:-4070}
SPEC_TOKENS=${SPEC_TOKENS:-11}         # repeated MTP runtime steps / draft predictions
BUFFER_GB=${BUFFER_GB:-6}              # dynamic-batching buffer; lower if OOM
MAX_REQUESTS=${MAX_REQUESTS:-16}       # max concurrent server requests; lower if OOM
OSL=${OSL:-1024}
PORT=${PORT:-5000}
PARSERS=${PARSERS:-deepseek-r1-reasoning qwen3-coder-tool}

# TP=1 parity run needs no forced CUDA connection limit.
if [ "${TP}" -gt 1 ]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
else
    unset CUDA_DEVICE_MAX_CONNECTIONS
fi

# Previous Nemotron-3-Super-120B model architecture args (kept for reference):
# MODEL_ARGS="\
#     --hidden-size 4096 --ffn-hidden-size 2688 --seq-length 1048576 \
#     --num-attention-heads 32 --num-query-groups 2 --group-query-attention --kv-channels 128 \
#     --max-position-embeddings 1048576 --position-embedding-type none \
#     --rotary-base 10000 --rotary-percent 1.0 --disable-bias-linear --squared-relu \
#     --untie-embeddings-and-output-weights --normalization RMSNorm \
#     --attention-dropout 0.0 --hidden-dropout 0.0 \
#     --mtp-hybrid-override-pattern none --mtp-use-repeated-layer \
#     --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
#     --num-experts 512 --moe-layer-freq 1 --moe-ffn-hidden-size 2688 --moe-router-topk 22 \
#     --moe-grouped-gemm --moe-shared-expert-intermediate-size 5376 \
#     --moe-router-score-function sigmoid --moe-router-enable-expert-bias --moe-router-topk-scaling-factor 5.0 \
#     --mamba-state-dim 128 --mamba-head-dim 64 --mamba-num-groups 8 --mamba-num-heads 128 \
#     --hybrid-layer-pattern 'MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME/*E/*E' \
#     --moe-latent-size 1024 --padded-vocab-size 131072 --model-provider hybrid \
#     --inference-max-seq-length ${MAX_SEQ_LENGTH:-16384}"

# NanoV3 3B MTP model architecture. The base architecture comes from
# 3B_nanov3_nebius_mtp_2_repeated.sh; the 128K lengths and unified MTP pattern
# come from the target iter_0004070 checkpoint metadata.
MODEL_ARGS="\
    --hidden-size 2688 --ffn-hidden-size 1856 --num-layers 54 \
    --seq-length 131072 --max-position-embeddings 131072 \
    --num-attention-heads 32 --num-query-groups 8 --group-query-attention --kv-channels 128 \
    --position-embedding-type none --disable-bias-linear --squared-relu \
    --use-fused-weighted-squared-relu \
    --untie-embeddings-and-output-weights --normalization RMSNorm \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --mtp-use-repeated-layer \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --num-experts 128 --moe-layer-freq 1 --moe-ffn-hidden-size 1856 --moe-router-topk 6 \
    --moe-grouped-gemm --moe-shared-expert-intermediate-size 3712 \
    --moe-router-score-function sigmoid --moe-router-enable-expert-bias --moe-router-topk-scaling-factor 2.5 \
    --mamba-state-dim 128 --mamba-head-dim 64 --mamba-num-groups 8 --mamba-num-heads 64 \
    --hybrid-layer-pattern 'MEMEM*EMEM*EMEM*EMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEME/*E/*E' \
    --padded-vocab-size 131072 --model-provider mamba \
    --inference-max-seq-length ${MAX_SEQ_LENGTH:-16384}"

# Previous Nemotron-3-Super-120B server/inference args (kept for reference):
# SERVER_ARGS="\
#     --micro-batch-size 1 --bf16 --te-rng-tracker --inference-rng-tracker \
#     --tensor-model-parallel-size ${TP} --expert-model-parallel-size ${EP} --expert-tensor-parallel-size 1 \
#     --pipeline-model-parallel-size 1 --sequence-parallel \
#     --load ${CKPT} --use-checkpoint-args --dist-ckpt-strictness log_unexpected \
#     --tokenizer-type HuggingFaceTokenizer --tokenizer-model ${TOKENIZER} --no-use-tokenizer-model-from-checkpoint-args \
#     --moe-router-dtype fp32 --moe-token-dispatcher-type alltoall --moe-permute-fusion \
#     --attention-backend flash --transformer-impl inference_optimized \
#     --inference-grouped-gemm-backend vllm --inference-use-synchronous-zmq-collectives --moe-shared-expert-overlap \
#     --inference-dynamic-batching --inference-dynamic-batching-unified-memory-level 0 \
#     --inference-dynamic-batching-max-tokens 2048 --inference-dynamic-batching-mamba-memory-ratio 0.21 \
#     --enable-chunked-prefill --use-flashinfer-fused-rope \
#     --inference-dynamic-batching-buffer-size-gb ${BUFFER_GB} --inference-dynamic-batching-max-requests ${MAX_REQUESTS} \
#     --inference-dynamic-batching-num-cuda-graphs -1 --cuda-graph-impl local --inference-cuda-graph-scope block \
#     --inference-logging-step-interval 1000 \
#     --parsers ${PARSERS} \
#     --host 0.0.0.0 --port ${PORT}"

# NanoV3 3B server/inference args. The checkpoint was saved at iter_0004070
# without a latest_checkpointed_iteration.txt tracker, so ckpt-step is explicit.
# Chunked prefill stays disabled to match the vLLM reference and let the
# attention-based repeated-MTP cache consume each complete prompt prefill.
SERVER_ARGS="\
    --micro-batch-size 1 --bf16 --te-rng-tracker --inference-rng-tracker \
    --tensor-model-parallel-size ${TP} --expert-model-parallel-size ${EP} --expert-tensor-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --load ${CKPT} --ckpt-step ${CKPT_STEP} --use-checkpoint-args \
    --dist-ckpt-strictness log_unexpected --ckpt-format torch_dist \
    --ckpt-fully-parallel-load --ckpt-assume-constant-structure --no-load-optim \
    --tokenizer-type SFTTokenizer --sft-tokenizer-prompt-format identity \
    --tokenizer-model ${TOKENIZER} --no-use-tokenizer-model-from-checkpoint-args \
    --moe-router-dtype fp32 --moe-token-dispatcher-type alltoall --moe-permute-fusion \
    --attention-backend flash --transformer-impl inference_optimized \
    --mamba-inference-conv-states-dtype fp32 --mamba-inference-ssm-states-dtype fp32 \
    --inference-moe-token-dispatcher-type nvls \
    --inference-use-synchronous-zmq-collectives --moe-shared-expert-overlap \
    --inference-dynamic-batching --inference-dynamic-batching-unified-memory-level 0 \
    --inference-dynamic-batching-max-tokens ${MAX_SEQ_LENGTH:-16384} --inference-dynamic-batching-mamba-memory-ratio 0.21 \
    --inference-dynamic-batching-buffer-size-gb ${BUFFER_GB} --inference-dynamic-batching-max-requests ${MAX_REQUESTS} \
    --cuda-graph-impl none --inference-cuda-graph-scope none \
    --inference-logging-step-interval 1000 \
    --parsers ${PARSERS} \
    --host 0.0.0.0 --port ${PORT}"

# MTP speculative decoding. The checkpoint owns the physical layer count;
# num-speculative-tokens invokes its repeated layer for every draft prediction.
SPEC_ARGS=""
if [ "${SPEC_TOKENS}" -gt 0 ]; then
    SPEC_ARGS="--num-speculative-tokens ${SPEC_TOKENS}"
fi

SERVER_CMD="python -m tools.run_dynamic_text_generation_server ${MODEL_ARGS} ${SERVER_ARGS} ${SPEC_ARGS}"

# --- Launch the server (every rank) -----------------------------------------
if [ "${RANK}" == "0" ]; then
    { echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"; echo "WORLD_SIZE=${WORLD_SIZE}"; \
      echo "CKPT=${CKPT}"; echo "TOKENIZER=${TOKENIZER}"; echo "SPECDEC_RUNTIME_PARAMS=${SPECDEC_RUNTIME_PARAMS}"; \
      echo "TP=${TP} EP=${EP}"; \
      echo "SPEC_TOKENS=${SPEC_TOKENS} OSL=${OSL} CONCURRENCY=${CONCURRENCY:-16} MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384} BUFFER_GB=${BUFFER_GB} MAX_REQUESTS=${MAX_REQUESTS}"; \
      echo "CMD=${SERVER_CMD}"; } > "${EXP_DIR}/config.env"
fi

echo "[rank ${RANK}] ${SERVER_CMD}"
eval "${SERVER_CMD}" > "${EXP_DIR}/server_rank${RANK}.log" 2>&1 &
SERVER_PID=$!

# --- Rank 0: wait for readiness, run the client, tear down ------------------
if [ "${RANK}" == "0" ]; then
    echo "Waiting for server to be ready (this includes checkpoint load; can take ~10 min)..."
    until grep -q "Running on http://0.0.0.0:${PORT}" "${EXP_DIR}/server_rank0.log" 2>/dev/null; do
        # Bail out early if the server process died during init (e.g. OOM).
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: server process exited before becoming ready. See ${EXP_DIR}/server_rank0.log" >&2
            echo "failed" > "${EXP_DIR}/done_status"
            exit 1
        fi
        sleep 5
    done
    echo "Server ready. Running specdec_bench acceptance-rate client..."

    SPECDEC_BENCH=${SPECDEC_BENCH:?set SPECDEC_BENCH to your specdec_bench checkout}
    # SAVE_DIR must be absolute (client runs from a `cd` subshell).
    SAVE_DIR=${SAVE_DIR:-$(pwd)/${EXP_DIR}/ar_results}
    mkdir -p "${SAVE_DIR}"
    # Client deps (the server ranks already have torch/TE): specdec_bench's
    # pinned requirements + aiohttp for the async HTTP client.
    pip install -r "${SPECDEC_BENCH}/requirements.txt" aiohttp

    EXTRA_ARGS=""
    [ -n "${NUM_REQUESTS:-}" ] && EXTRA_ARGS="${EXTRA_ARGS} --num_requests ${NUM_REQUESTS}"
    [ -n "${SPEED_BENCH_CATEGORY:-}" ] && EXTRA_ARGS="${EXTRA_ARGS} --category ${SPEED_BENCH_CATEGORY}"
    [ -n "${SPECDEC_RUNTIME_PARAMS:-}" ] && EXTRA_ARGS="${EXTRA_ARGS} --runtime_params ${SPECDEC_RUNTIME_PARAMS}"

    ( cd "${SPECDEC_BENCH}" && python3 -u run.py \
        --engine MEGATRON \
        --base_url "http://localhost:${PORT}" \
        --model_dir "${TOKENIZER}" \
        --dataset speed \
        --dataset_path "${SPEED_BENCH_DATA}" \
        --tokenizer "${TOKENIZER}" \
        --speculative_algorithm MTP \
        --draft_length "${SPEC_TOKENS}" \
        --output_length "${OSL}" \
        --concurrency "${CONCURRENCY:-16}" \
        --tp_size "${TP}" \
        --ep_size "${EP}" \
        --show_progress \
        --save_dir "${SAVE_DIR}" \
        ${EXTRA_ARGS} ) 2>&1 | tee "${EXP_DIR}/ar_benchmark.log"
    CLIENT_STATUS=${PIPESTATUS[0]}

    if [ "${CLIENT_STATUS}" -ne 0 ]; then
        echo "failed" > "${EXP_DIR}/done_status"
        echo "ERROR: specdec_bench exited with status ${CLIENT_STATUS}" >&2
        kill "${SERVER_PID}" 2>/dev/null
        wait "${SERVER_PID}" 2>/dev/null
        exit "${CLIENT_STATUS}"
    fi

    echo "done" > "${EXP_DIR}/done_status"
    echo "Results in: ${SAVE_DIR}"
else
    # Non-zero ranks: keep the server alive until rank 0 signals completion.
    until [ -f "${EXP_DIR}/done_status" ]; do sleep 30; done
fi

# Gracefully stop the server.
kill "${SERVER_PID}" 2>/dev/null
wait "${SERVER_PID}" 2>/dev/null
