#!/usr/bin/env bash
set -euo pipefail

PYTHON=.venv/bin/python
ACCELERATE=.venv/bin/accelerate

evaluate() {
  local config=$1
  local first=$2
  local second=$3
  local third=$4
  local fourth=$5
  local fifth=$6
  local name
  name=$(basename "$config" .yaml)

  (
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" evaluation/run.py "$first" --config "$config" \
      > "evaluation/logs/${name}-${first}.log" 2>&1
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" evaluation/run.py "$third" --config "$config" \
      > "evaluation/logs/${name}-${third}.log" 2>&1
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" evaluation/run.py "$fifth" --config "$config" \
      > "evaluation/logs/${name}-${fifth}.log" 2>&1
  ) &
  local gpu0=$!

  (
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" evaluation/run.py "$second" --config "$config" \
      > "evaluation/logs/${name}-${second}.log" 2>&1
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" evaluation/run.py "$fourth" --config "$config" \
      > "evaluation/logs/${name}-${fourth}.log" 2>&1
  ) &
  local gpu1=$!

  wait "$gpu0"
  wait "$gpu1"
  "$PYTHON" evaluation/summarize.py --config "$config"
}

mkdir -p logs evaluation/logs

"$ACCELERATE" launch --num_processes 2 train.py --config configs/experiment5.yaml \
  > logs/experiment5.log 2>&1
evaluate evaluation/experiment5.yaml \
  base checkpoint-82 checkpoint-164 checkpoint-246 final

"$ACCELERATE" launch --num_processes 2 train.py --config configs/experiment6.yaml \
  > logs/experiment6.log 2>&1
evaluate evaluation/experiment6.yaml \
  base checkpoint-285 checkpoint-570 checkpoint-855 final
