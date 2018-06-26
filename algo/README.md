# study-ppo

(ikostrikov) tor@l7480:~/ws/pytorch-a2c-ppo-acktr$
python main.py --env-name "Reacher-v2" --algo ppo --use-gae --vis-interval 1  --log-interval 1 --num-stack 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-frames 1000000 --no-vis


python main.py
--env-name "Reacher-v2"
--algo ppo
--use-gae

--num-stack 1
--num-steps 2048
--num-processes 1

--lr 3e-4
--entropy-coef 0
--value-loss-coef 1
--ppo-epoch 10
--num-mini-batch 32
--gamma 0.99
--tau 0.95
--num-frames 1000000

--vis-interval 1
--log-interval 1
--no-vis

## question
* why hardcode `torch.set_num_threads(.)` @main.py?

## log
* 20180626
```
commit 31ca5468d85961ed772324ccfcc477e66ea74c54
Author: Vektor Dewanto <vektor.dewanto@gmail.com>
Date:   Tue Jun 26 10:50:15 2018 +1000

(ikostrikov) tor@l7480:~/ws/pytorch-a2c-ppo-acktr$ python main.py --env-name "Reacher-v2" --algo ppo --use-gae --vis-interval 1  --log-interval 1 --num-stack 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 1 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-frames 1000000

Updates 480/487, num timesteps 985088, FPS 1103, mean/median reward -0.5/-0.5, min/max reward -0.5/-0.5, entropy -3.83383, value loss 0.00016, policy loss 0.00580
Updates 481/487, num timesteps 987136, FPS 1103, mean/median reward -0.1/-0.1, min/max reward -0.1/-0.1, entropy -3.81993, value loss 0.00011, policy loss 0.00439
Updates 482/487, num timesteps 989184, FPS 1103, mean/median reward -0.5/-0.5, min/max reward -0.5/-0.5, entropy -3.81405, value loss 0.00011, policy loss 0.00691
Updates 483/487, num timesteps 991232, FPS 1103, mean/median reward -0.4/-0.4, min/max reward -0.4/-0.4, entropy -3.80114, value loss 0.00012, policy loss 0.00130
Updates 484/487, num timesteps 993280, FPS 1103, mean/median reward -0.3/-0.3, min/max reward -0.3/-0.3, entropy -3.79238, value loss 0.00015, policy loss 0.00437
Updates 485/487, num timesteps 995328, FPS 1103, mean/median reward -0.1/-0.1, min/max reward -0.1/-0.1, entropy -3.79141, value loss 0.00008, policy loss 0.00369
Updates 486/487, num timesteps 997376, FPS 1103, mean/median reward -0.2/-0.2, min/max reward -0.2/-0.2, entropy -3.79600, value loss 0.00014, policy loss 0.00239
Updates 487/487, num timesteps 999424, FPS 1103, mean/median reward -0.3/-0.3, min/max reward -0.3/-0.3, entropy -3.80010, value loss 0.00013, policy loss 0.00623
```
