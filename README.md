# study-ppo

## fact
* the return modulator in the loss, psi, is set to the advantage, A = Q - V
* filter both reward and observ using VecNormalize()
* do clip the gradient
```
nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                         self.max_grad_norm)
```
* use state value, NOT action-state value
* reset is NOT called during rollout;
  * this is NOT similar with that of openai-baselines
  * in /home/tor/ws/baselines/baselines/acktr/acktr_cont_kfac.py
    * reset is at every beginning of run_one_episode()
* NOT use concat_observ
* plot return vs nstep, using
  * smothing: smooth_reward_curve(x, y)
  * fix_point(x, y, interval)
* most params are shared between both actor and critic nets
* use Monitor():
  /home/tor/ws/poacp/xternal/baselines/baselines/bench/monitor.py
  * print if done==True
```py
eprew = sum(self.rewards)
eplen = len(self.rewards)
epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
```

## question
* why set return[-1]=next_value?
  why not set the last return to be (0 if done, else copy from the last value), like:
  `vpred_t = np.append(vpred_t, 0.0 if path["terminated"] else vpred_t[-1])` at
  https://github.com/openai/baselines/blob/f2729693253c0ef4d4086231d36e0a4307ec1cb3/baselines/acktr/acktr_cont.py#L102
```py
def compute_returns(self, next_value, use_gae, gamma, tau):
    ...
    else:
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * \
                gamma * self.masks[step + 1] + self.rewards[step]
```
* why this becomes old_action_log_probs?
  there is not update yet, isnt?
```
data_generator = rollouts.feed_forward_generator(
advantages, self.num_mini_batch)

for sample in data_generator:
    observations_batch, states_batch, actions_batch, \
       return_batch, masks_batch, old_action_log_probs_batch, \
            adv_targ = sample
```

## question: answered
* why act() returns pred_state_value, in addition to act and act_log_prob:
  `action, action_log_prob, pred_state_value = actor_critic_net.act(observ)`
  * it is used to compute advantage:
    `pred_advs = rollouts.returns[:-1] - rollouts.pred_state_values[:-1]`
* states? cf observation
  * seems only for atari, or image inputs
* max_grad_norm?
  * for clipping the grad, before optim.step()
* why adv computed this way?
  Q from empirical;  V from prediction
  * thus, we have predicted advantage, only Q can be obtained empirically
  * true V is expectation over all actions
```py
def update(self, rollouts, eps=1e-5):
    # Compute advantages: $A(s_t, a_t) = Q(s_t, a_t) - V(s_t, a_t)$
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
```
* what does this do? from openai-baselines:
  `envs = VecNormalize(envs, gamma=args.gamma)`
  * normalize and clip observ and reward
  * filter observ
```py
def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
    VecEnvWrapper.__init__(self, venv)
    self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
    self.ret_rms = RunningMeanStd(shape=()) if ret else None
```
  * this VecNormalize does not allow reset at every episode, unless
    the env is wrapped with Monitor(allow_early_reset=True)
* `mask` used for?
  * masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
  * needed because rollout is based on nstep,
    not using the notion of episode based on done
  * there are multiple process that may have different episode length
* num_steps? for learning batch?
  * nstep per update
  * see: num_updates = int(args.num_frames) // args.num_steps // args.num_processes

## todo
* reset per episode
* using entropy info of action distrib
* nprocess > 1
* use GAE
* recurrent net
* gym robotic env
* cuda compatibility

## setup
* visdom
  * pip install visdom
  * python -m visdom.server
