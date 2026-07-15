#!/bin/bash
# SPEED-Bench MTP acceptance-rate benchmark -- SLURM entry point.
#
# Launches a Megatron dynamic inference server (with MTP speculative decoding)
# and the specdec_bench acceptance-rate client in a single allocation. Run from
# the root of your Megatron-LM checkout:
#
#   SPEC_TOKENS=2 CKPT=... TOKENIZER=... SPECDEC_BENCH=... SPEED_BENCH_DATA=... \
#     sbatch --gres=gpu:8 scripts/speed_bench/run_speed_bench.slurm
#
# SBATCH resource lines below are DEFAULTS; override on the command line, e.g.
#   sbatch -A my_account -p interactive -q normal --gres=gpu:8 scripts/speed_bench/run_speed_bench.slurm
# See scripts/speed_bench/README.md for the full knob list and troubleshooting.
#SBATCH --job-name=speed_bench
# Previous default partition:
##SBATCH -p batch
# Default all future SPEED-Bench submissions to the interactive partition.
#SBATCH -p interactive
#SBATCH -q normal
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
# NOTE: supply a SLURM account via `sbatch -A <account> ...` or the
# SBATCH_ACCOUNT env var. Partition/QOS above (interactive/normal) are examples --
# override with `-p`/`-q` for your cluster.

set -uo pipefail

# --- Cluster / infra knobs (env-overridable) --------------------------------
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}          # must match --gres=gpu:N
NUM_GPUS=$(( NUM_GPUS_PER_NODE * NUM_NODES ))
CONTAINER_IMAGE=${CONTAINER_IMAGE:?set CONTAINER_IMAGE to your mcore-inference container (.sqsh)}
# Mounts: /lustre plus /home (the example checkpoint lives under /home, outside
# /lustre). Add more off-filesystem paths via EXTRA_MOUNTS (comma-sep src:dst).
EXTRA_MOUNTS=${EXTRA_MOUNTS:-/home:/home}
CONTAINER_MOUNTS="/lustre:/lustre${EXTRA_MOUNTS:+,${EXTRA_MOUNTS}}"
WORKDIR="${PWD}"                                   # the Megatron-LM checkout root

# --- Benchmark knobs (exported so srun propagates them to every rank) -------
export SPEC_TOKENS=${SPEC_TOKENS:-11}
export CKPT=${CKPT:?set CKPT to your Megatron MTP checkpoint directory}
export TOKENIZER=${TOKENIZER:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}
# Previous Nemotron-3-Super inference defaults:
# export TP=${TP:-2}
# export EP=${EP:-8}
# NanoV3 3B single-node inference defaults:
export TP=${TP:-1}
export EP=${EP:-8}
export CKPT_STEP=${CKPT_STEP:-4070}
export BUFFER_GB=${BUFFER_GB:-6}
export MAX_REQUESTS=${MAX_REQUESTS:-16}
export OSL=${OSL:-1024}
export CONCURRENCY=${CONCURRENCY:-16}
export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-16384}
export NUM_REQUESTS=${NUM_REQUESTS:-}                # empty = full dataset; set only for smoke tests
export SPEED_BENCH_CATEGORY=${SPEED_BENCH_CATEGORY:-}
export SPECDEC_BENCH=${SPECDEC_BENCH:?set SPECDEC_BENCH to your specdec_bench checkout}
export SPECDEC_RUNTIME_PARAMS=
export SPEED_BENCH_DATA=${SPEED_BENCH_DATA:?set SPEED_BENCH_DATA to your staged SPEED-Bench parquet dir}
export AA_DATA=${AA_DATA:-/lustre/fsw/portfolios/coreai/users/iputterman/aa_gen_16k_1k.jsonl}
export SAVE_DIR=${SAVE_DIR:-}                      # defaults to <EXP_DIR>/ar_results
# HF cache (where the tokenizer is / will be cached).
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}

EXPECTED_WORLD_SIZE=$(( TP * EP ))
if [ "${NUM_GPUS}" -ne "${EXPECTED_WORLD_SIZE}" ]; then
    echo "ERROR: TP=${TP} and EP=${EP} require ${EXPECTED_WORLD_SIZE} ranks, but allocation has ${NUM_GPUS}." >&2
    exit 2
fi

# --- Experiment directory ---------------------------------------------------
export EXP_NAME=${EXP_NAME:-speed_bench}
RESULTS_DIR=${RESULTS_DIR:-scripts/speed_bench/experiments}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export EXP_DIR="${RESULTS_DIR}/${TIMESTAMP}_${EXP_NAME}"
mkdir -p "${EXP_DIR}"
echo "Experiment directory: ${EXP_DIR}"

# --- CPU allocation --------------------------------------------------------
# One task per GPU. Derive cpus-per-task from what SLURM actually granted on the
# node so the step never over-requests CPUs (nodes vary; a fixed value fails on
# smaller nodes with "More processors requested than permitted").
NTASKS_PER_NODE=${NUM_GPUS_PER_NODE}
_alloc_cpus=${SLURM_CPUS_ON_NODE:-128}
CPUS_PER_TASK=$(( _alloc_cpus / NUM_GPUS_PER_NODE ))
[ "${CPUS_PER_TASK}" -lt 1 ] && CPUS_PER_TASK=1

# --- Launch -----------------------------------------------------------------
srun -l \
    --nodes="${NUM_NODES}" \
    --ntasks="${NUM_GPUS}" \
    --ntasks-per-node="${NTASKS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir="${WORKDIR}" \
    bash scripts/speed_bench/speed_bench_worker.sh
