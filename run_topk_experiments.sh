#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-halfcheetah-medium-replay-v2}"
SEEDS="${SEEDS:-0 1 2}"
COMMON_ARGS="--env_name ${ENV_NAME} --training_steps 1000000 --eval_every 10000"

for SEED in ${SEEDS}; do
  python train.py ${COMMON_ARGS} --seed "${SEED}" --mode_selection top1 --group top1
  python train.py ${COMMON_ARGS} --seed "${SEED}" --mode_selection topk_q --top_k 3 --group topk_q
  python train.py ${COMMON_ARGS} --seed "${SEED}" --mode_selection topk_gnn --top_k 3 --group topk_gnn
  python train.py ${COMMON_ARGS} --seed "${SEED}" --mode_selection topk_gnn_merge --top_k 3 --group topk_gnn_merge
done
