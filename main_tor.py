#!/usr/bin/env python3
import sys

import torch
import numpy as np
from visdom import Visdom

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

import algo
from envs import make_env
from model import Policy
from model_tor import ActorCriticNetwork
from storage import RolloutStorage

def main():
    if len(sys.argv)!=2:
        print('Wrong argv!')
        return
    nupdate = int(sys.argv[1])

    # Init
    viz = Visdom(port=8097)
    xprmt_dir = '/home/tor/xprmt/ikostrikov2'
    env_id = 'Reacher-v2'
    nprocess = 1
    nstack = 1
    nstep = 2500
    gamma = 0.99
    eps = 1e-5
    seed = 123
    log_interval = 1
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    assert nprocess==1
    assert nstack==1
    # assert not using cuda!

    envs = [make_env(env_id, seed=seed, rank=i, log_dir=xprmt_dir, add_timestep=False)
            for i in range(nprocess)]
    envs = DummyVecEnv(envs)
    envs = VecNormalize(envs, ob=True, ret=True, gamma=gamma, epsilon=eps, clipob=10., cliprew=10.)
    assert len(envs.observation_space.shape)==1
    assert len(envs.action_space.shape)==1
    assert envs.action_space.__class__.__name__ == "Box"
    observ_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]

    actor_critic_net = Policy(envs.observation_space.shape, envs.action_space, recurrent_policy=False)
    # actor_critic_net = ActorCriticNetwork(input_dim=observ_dim,
    #                                       actor_output_dim=action_dim,
    #                                       critic_output_dim=1)

    rollouts = RolloutStorage(nstep, nprocess, envs.observation_space.shape,
                              envs.action_space, actor_critic_net.state_size)

    agent = algo.PPO(actor_critic_net, clip_param=0.2, ppo_epoch=10, num_mini_batch=32,
                     value_loss_coef=1.0, entropy_coef=0.0,
                     lr=3e-4, eps=eps, max_grad_norm=0.5)

    # Learning
    observ = envs.reset()
    observ = torch.from_numpy(observ).float()
    rollouts.observations[0].copy_(observ)

    for update_idx in range(nupdate):
        # Rollout
        for step_idx in range(nstep):
            # Sample actions
            with torch.no_grad():
                 act_resp = actor_critic_net.act(rollouts.observations[step_idx],
                                                    rollouts.states[step_idx],
                                                    rollouts.masks[step_idx])
                 value, action, action_log_prob, state = act_resp

            print(value)
            print(action)
            print(action_log_prob)
            print(state)
            exit()

            # Step
            observ, reward, done, info = envs.step(action.squeeze(1).cpu().numpy())

            mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            observ = torch.from_numpy(observ).float()
            observ *= mask

            rollouts.insert(observ, state, action, action_log_prob, value, reward, mask)

        # Update
        with torch.no_grad():
            next_value = actor_critic_net.get_value(rollouts.observations[-1],
                                                    rollouts.states[-1],
                                                    rollouts.masks[-1])
            next_value = next_value.detach()

        rollouts.compute_returns(next_value, gamma=gamma, use_gae=False, tau=None)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # Log
        if update_idx % log_interval == 0:
            total_nstep = (update_idx+1) * nprocess * nstep
            logs  = ['update {}/{}'.format(update_idx+1, nupdate)]
            logs += ['nstep {}'.format(total_nstep)]
            logs += ['action_loss {:.5f}'.format(action_loss)]
            logs += ['value_loss {:.5f}'.format(value_loss)]
            logs += ['dist_entropy {:.5f}'.format(dist_entropy)]
            print(' | '.join(logs))

if __name__ == '__main__':
    main()
