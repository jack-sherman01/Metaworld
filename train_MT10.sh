#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hzhang/heng/omniR/Metaworld"
cd "$ROOT"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/MT10/$RUN_ID"
MODEL_BASE_DIR="$ROOT/trained_models"
mkdir -p "$LOG_DIR"

echo "Training all MT10 tasks..."

PYTHONUNBUFFERED=1 python -u scripts/train_sb3_single_task.py \
  --benchmark MT10 \
  --all-tasks \
  --algo sac \
  --total-timesteps 1000000 \
  --n-envs 8 \
  --reward-version v2 \
  --save-root "$MODEL_BASE_DIR" \
  2>&1 | tee "$LOG_DIR/all_tasks.log"

echo "All tasks completed!"