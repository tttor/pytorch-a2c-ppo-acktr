# study-ppo

## question
* what does this do? from openai-baselines
  `envs = VecNormalize(envs, gamma=args.gamma)`
* max_grad_norm?

* current_obs vs obs?
* `mask` used for?
  * masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
  * needed because rollout is based on nstep,
    not using the notion of episode based on done
  * there are multiple process that may have different episode length
* num_steps? for learning batch?
  * nstep per update
  * see: num_updates = int(args.num_frames) // args.num_steps // args.num_processes

## setup
* visdom
  * pip install visdom
  * python -m visdom.server
