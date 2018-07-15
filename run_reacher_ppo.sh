#!/bin/bash

python main.py \
--env-name "Reacher-v2" \
--algo ppo \
--use-gae \
--num-stack 1 \
--num-steps 2048 \
--num-processes 1 \
--lr 3e-4 \
--entropy-coef 0 \
--value-loss-coef 1 \
--ppo-epoch 10 \
--num-mini-batch 32 \
--gamma 0.99 \
--tau 0.95 \
--num-frames 1000000 \
--vis-interval 1 \
--log-interval 1
