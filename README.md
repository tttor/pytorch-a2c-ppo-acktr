# study-ppo

## question
* max_grad_norm?
* states? cf observation
* current_obs vs obs?
* random seed does not control gym?
* continue from where last ep end, instead of reset during rollout?

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

## fact
* most params are shared between both actor and critic nets
* NOT use concat_observ
* use Monitor():
  /home/tor/ws/poacp/xternal/baselines/baselines/bench/monitor.py
  * print if done==True
```py
eprew = sum(self.rewards)
eplen = len(self.rewards)
epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
```
* reset is not called during rollout;
  * this is NOT similar with that of openai-baselines
  * in /home/tor/ws/baselines/baselines/acktr/acktr_cont_kfac.py
    * reset is at every beginning of run_one_episode()
* filter both reward and observ using VecNormalize()

## setup
* visdom
  * pip install visdom
  * python -m visdom.server
