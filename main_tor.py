#!/usr/bin/env python3
import os
import sys
import socket
import datetime
import argparse
import torch
import numpy as np
from baselines import logger
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model_tor import ActorCriticNetwork
from storage_tor import ExperienceBuffer
from ppo_tor import VanillaPPO

def main():
    # Init
    args = parse_args()
    env_id = 'Reacher-v2'
    nprocess = 1
    nstep_per_update = 2500
    gamma = 0.99
    epsilon = 1e-5
    log_interval = 1
    use_gae=False; tau=None
    tag = '_'.join(['ppo', env_id, args.opt])
    log_dir = os.path.join(args.log_dir, make_stamp(tag))
    logger.configure(dir=log_dir)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    assert nprocess==1
    # assert not using cuda!
    # assert not using recurrent net!

    ppo_clip_eps = 0.2
    ppo_max_grad_norm = 0.5
    ppo_optim_id = args.opt
    ppo_lr = 3e-4
    ppo_nepoch = 10
    ppo_nminibatch = 32

    envs = [make_env(env_id, seed=args.seed, rank=i, log_dir=log_dir, add_timestep=False) for i in range(nprocess)]
    envs = DummyVecEnv(envs)
    envs = VecNormalize(envs, ob=True, ret=True, gamma=gamma, epsilon=epsilon, clipob=10., cliprew=10.)
    observ_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]
    assert len(envs.observation_space.shape)==1
    assert len(envs.action_space.shape)==1
    assert envs.action_space.__class__.__name__ == "Box"

    actor_critic_net = ActorCriticNetwork(input_dim=observ_dim,
                                          hidden_dim=64,
                                          actor_output_dim=action_dim,
                                          critic_output_dim=1)
    rollouts = ExperienceBuffer(nstep_per_update, nprocess, observ_dim, action_dim)
    agent = VanillaPPO(actor_critic_net, ppo_clip_eps, ppo_max_grad_norm,
                       ppo_optim_id, ppo_lr, ppo_nepoch, ppo_nminibatch, epsilon)


    # Learning
    observ = envs.reset()
    observ = torch.from_numpy(observ).float()
    rollouts.observations[0].copy_(observ)

    for update_idx in range(args.n_update):
        # Rollout for nstep_per_update steps
        for step_idx in range(nstep_per_update):
            # Sample actions
            with torch.no_grad():
                action, action_log_prob, pred_state_value = actor_critic_net.act(observ)
            # print(action); print(action_log_prob); print(pred_state_value)
            # exit()

            # Step
            observ, reward, done, info = envs.step(action.squeeze(1).cpu().numpy())

            mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            observ = torch.from_numpy(observ).float()
            observ *= mask

            rollouts.insert(action, action_log_prob, pred_state_value, reward, next_observ=observ, next_mask=mask)

        # Prepare for update
        with torch.no_grad():
            pred_next_state_value = actor_critic_net.predict_state_value(observ).detach()
        rollouts.compute_returns(pred_next_state_value, gamma)

        # Update
        loss, value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # Log
        if (update_idx % log_interval)==0:
            nstep_so_far = (update_idx+1) * nprocess * nstep_per_update
            logs  = ['update {}/{}'.format(update_idx+1, args.n_update)]
            logs += ['loss {:.5f}'.format(loss)]
            logs += ['action_loss {:.5f}'.format(action_loss)]
            logs += ['value_loss {:.5f}'.format(value_loss)]
            logs += ['dist_entropy {:.5f}'.format(dist_entropy)]
            logs += ['nstep_so_far {}'.format(nstep_so_far)]
            logger.log(' | '.join(logs))

def make_stamp(tag):
    hostname = socket.gethostname(); hostname = hostname.split('.')[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stamp = '_'.join([tag, hostname, timestamp])
    return stamp

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--opt', help='optimizer ID', type=str, default=None, required=True)
    parser.add_argument('--n_update', help='number of update', type=int, default=None, required=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None, required=True)
    parser.add_argument('--log_dir', help='root xprmt log dir', type=str, default=None, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main()
