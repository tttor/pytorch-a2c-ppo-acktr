#!/usr/bin/env python3
import torch
from visdom import Visdom

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

import algo
from envs import make_env
from model import Policy
from storage import RolloutStorage

def main():
    # Init
    torch.set_num_threads(4)
    viz = Visdom(port=8097)
    xprmt_dir = '/home/tor/xprmt/ikostrikov2'
    nprocess = 1
    nstack = 1
    nstep = 2048
    gamma = 0.99
    eps = 1e-5
    seed = 123

    envs = [make_env('Reacher-v2', seed=seed, rank=i, log_dir=xprmt_dir, add_timestep=False)
            for i in range(nprocess)]
    envs = DummyVecEnv(envs)
    envs = VecNormalize(envs, gamma=gamma)
    assert nprocess==1
    assert len(envs.observation_space.shape)==1
    assert envs.action_space.__class__.__name__ == "Box"

    policy = Policy(envs.observation_space.shape, envs.action_space, recurrent_policy=False)

    agent = algo.PPO(policy, clip_param=0.2, ppo_epoch=10, num_mini_batch=32,
                     value_loss_coef=1.0, entropy_coef=0.0,
                     lr=3e-4, eps=eps, max_grad_norm=0.5)

    rollouts = RolloutStorage(nstep, nprocess, envs.observation_space.shape,
                              envs.action_space, policy.state_size)

if __name__ == '__main__':
    main()
