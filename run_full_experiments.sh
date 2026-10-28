#!/usr/bin/env bash
set -euo pipefail

PYTHON=.venv/bin/python
ACCELERATE=.venv/bin/accelerate

run_evaluations() {
  local config=$1
  local first_gpu0=$2
  local second_gpu0=$3
  local first_gpu1=$4
  local second_gpu1=$5
  local prefix
  prefix=$(basename "$config" .yaml)

  (
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" evaluation/run.py "$first_gpu0" --config "$config" \
      > "evaluation/logs/${prefix}-${first_gpu0}.log" 2>&1
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" evaluation/run.py "$second_gpu0" --config "$config" \
      > "evaluation/logs/${prefix}-${second_gpu0}.log" 2>&1
  ) &
  local gpu0=$!

  (
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" evaluation/run.py "$first_gpu1" --config "$config" \
      > "evaluation/logs/${prefix}-${first_gpu1}.log" 2>&1
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" evaluation/run.py "$second_gpu1" --config "$config" \
      > "evaluation/logs/${prefix}-${second_gpu1}.log" 2>&1
  ) &
  local gpu1=$!

  wait "$gpu0"
  wait "$gpu1"
  "$PYTHON" evaluation/summarize.py --config "$config"
}

mkdir -p logs evaluation/logs

"$ACCELERATE" launch --num_processes 2 train.py --config configs/experiment3.yaml \
  > logs/experiment3.log 2>&1
mkdir -p evaluation/results/experiment3/raw
cp -R evaluation/results/experiment2/raw/base evaluation/results/experiment3/raw/base
run_evaluations evaluation/experiment3.yaml \
  checkpoint-3000 checkpoint-9000 checkpoint-6000 final

"$ACCELERATE" launch --num_processes 2 train.py --config configs/experiment4.yaml \
  > logs/experiment4.log 2>&1
mkdir -p evaluation/results/experiment4/raw
cp -R evaluation/results/raw/base evaluation/results/experiment4/raw/base
run_evaluations evaluation/experiment4.yaml \
  checkpoint-5000 checkpoint-15000 checkpoint-10000 final
