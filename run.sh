#!/bin/bash
if [ "$#" -ne 2 ]; then
  echo "USAGE:"
  echo "bash run.sh <env_id> <log_dir>"
  exit 1
fi
env_id=${1}
log_dir=${2}

python main.py \
--algo ppo \
--lr 3e-4 \
--eps 1e-5 \
--gamma 0.99 \
--use-gae \
--tau 0.95 \
--entropy-coef 0.0 \
--value-loss-coef 1.0 \
--max-grad-norm 0.5 \
--seed 12 \
--num-processes 1 \
--num-steps 2048 \
--ppo-epoch 10 \
--num-mini-batch 32 \
--clip-param 0.2 \
--num-stack 1 \
--log-interval 1 \
--save-interval 10 \
--vis-interval 1 \
--num-frames 1000000 \
--env-name $env_id \
--log-dir $log_dir \
--save-dir $log_dir \
--no-cuda \
--no-vis
