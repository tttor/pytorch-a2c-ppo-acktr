#!/usr/bin/env python3
import sys

import torch
import numpy as np

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from envs import make_env

def main():
    if len(sys.argv)!=3:
        print('Wrong argv!')
        return
    nupdate = int(sys.argv[1])
    mode = sys.argv[2]

    # Init
    xprmt_dir = '/home/tor/xprmt/ikostrikov2'
    env_id = 'Reacher-v2'
    nprocess = 1
    nstack = 1
    nstep_per_update = 2500
    gamma = 0.99
    eps = 1e-5
    seed = 123
    log_interval = 1
    use_gae=False; tau=None
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    assert nprocess==1
    assert nstack==1
    # assert not using cuda!
    # assert not using recurrent net!

    ppo_value_loss_coef = 1.0
    ppo_entropy_coef = 0.0
    ppo_clip_eps = 0.2
    ppo_nepoch = 10
    ppo_nminibatch = 32
    ppo_lr = 3e-4
    ppo_max_grad_norm = 0.5

    envs = [make_env(env_id, seed=seed, rank=i, log_dir=xprmt_dir, add_timestep=False)
            for i in range(nprocess)]
    envs = DummyVecEnv(envs)
    envs = VecNormalize(envs, ob=True, ret=True, gamma=gamma, epsilon=eps, clipob=10., cliprew=10.)
    observ_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]
    assert len(envs.observation_space.shape)==1
    assert len(envs.action_space.shape)==1
    assert envs.action_space.__class__.__name__ == "Box"

    if mode=='ori':
        from model import Policy
        from storage import RolloutStorage
        import algo

        actor_critic_net = Policy(envs.observation_space.shape, envs.action_space,
                                    recurrent_policy=False)
        rollouts = RolloutStorage(nstep_per_update, nprocess, envs.observation_space.shape,
                                  envs.action_space, actor_critic_net.state_size)
        agent = algo.PPO(actor_critic_net, ppo_clip_eps,
                            ppo_nepoch, ppo_nminibatch,
                            ppo_value_loss_coef, ppo_entropy_coef,
                            ppo_lr, eps, ppo_max_grad_norm)
    elif mode=='tor':
        from model_tor import ActorCriticNetwork
        from storage_tor import ExperienceBuffer
        from ppo_tor import VanillaPPO

        actor_critic_net = ActorCriticNetwork(input_dim=observ_dim,
                                                actor_output_dim=action_dim,
                                                critic_output_dim=1)
        rollouts = ExperienceBuffer(nstep_per_update, nprocess, observ_dim, action_dim)
        agent = VanillaPPO(actor_critic_net, ppo_clip_eps, ppo_max_grad_norm,
                            ppo_lr, ppo_nepoch, ppo_nminibatch, eps)
    else:
        raise NotImplementedError

    # Learning
    observ = envs.reset()
    observ = torch.from_numpy(observ).float()
    rollouts.observations[0].copy_(observ)

    for update_idx in range(nupdate):
        # Rollout
        for step_idx in range(nstep_per_update):
            # Sample actions
            with torch.no_grad():
                if mode=='ori':
                    act_response = actor_critic_net.act(rollouts.observations[step_idx],
                                                        rollouts.states[step_idx],
                                                        rollouts.masks[step_idx])
                    pred_state_value, action, action_log_prob, state = act_response
                elif mode=='tor':
                    action, action_log_prob, pred_state_value = actor_critic_net.act(observ)
                else:
                    raise NotImplementedError

            # print(action)
            # print(action_log_prob)
            # print(pred_state_value)
            # exit()

            # Step
            observ, reward, done, info = envs.step(action.squeeze(1).cpu().numpy())

            mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            observ = torch.from_numpy(observ).float()
            observ *= mask

            if mode=='ori':
                rollouts.insert(observ, state, action, action_log_prob, pred_state_value, reward, mask)
            elif mode=='tor':
                rollouts.insert(action, action_log_prob, pred_state_value, reward, next_observ=observ, next_mask=mask)
            else:
                raise NotImplementedError

        # Prepare for update
        if mode=='ori':
            with torch.no_grad():
                pred_next_state_value = actor_critic_net.get_value(rollouts.observations[-1],
                                                                    rollouts.states[-1],
                                                                    rollouts.masks[-1]).detach()
            rollouts.compute_returns(pred_next_state_value, use_gae, gamma, tau)
        elif mode=='tor':
            with torch.no_grad():
                pred_next_state_value = actor_critic_net.predict_state_value(observ).detach()
            rollouts.compute_returns(pred_next_state_value, gamma)
        else:
            raise NotImplementedError

        # Update
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # Log
        if (update_idx % log_interval)==0:
            nstep_so_far = (update_idx+1) * nprocess * nstep_per_update
            logs  = ['update {}/{}'.format(update_idx+1, nupdate)]
            logs += ['action_loss {:.5f}'.format(action_loss)]
            logs += ['value_loss {:.5f}'.format(value_loss)]
            logs += ['dist_entropy {:.5f}'.format(dist_entropy)]
            logs += ['nstep_so_far {}'.format(nstep_so_far)]
            print(' | '.join(logs))

if __name__ == '__main__':
    main()
