#!/bin/bash

python main.py \
--env-name "Reacher-v2" \
--seed 123 \
--num-stack 1 \
--gamma 0.99 \
--eps 1e-5 \
--num-frames 1000000 \
--num-processes 1 \
--num-steps 2048 \
--algo ppo \
--clip-param 0.2 \
--ppo-epoch 10 \
--num-mini-batch 32 \
--value-loss-coef 1.0 \
--entropy-coef 0.0 \
--lr 3e-4 \
--max-grad-norm 0.5 \
--vis-interval 1 \
--log-dir /home/tor/xprmt/ikostrikov \
--log-interval 1
