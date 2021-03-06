import torch
import torch.nn.functional as fn

class VanillaPPO():
    def __init__(self, actor_critic_net, clip_eps, max_grad_norm, optim_id, lr, n_epoch, n_minibatch, epsilon):
        self.actor_critic_net = actor_critic_net
        self.clip_eps = clip_eps
        self.n_epoch = n_epoch
        self.n_minibatch = n_minibatch
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        if optim_id=='adam':
            self.optim = torch.optim.Adam(actor_critic_net.parameters(), lr=lr, eps=epsilon)
        elif optim_id=='rmsprop':
            self.optim = torch.optim.RMSprop(actor_critic_net.parameters(), lr=lr, eps=epsilon)
        elif optim_id=='sgd':
            self.optim = torch.optim.SGD(actor_critic_net.parameters(), lr=lr)
        elif optim_id=='lbfgs':
            self.optim = torch.optim.LBFGS(actor_critic_net.parameters(), lr=lr)
        else:
            raise NotImplementedError

    def update(self, experience):
        # Compute advantages: $A(s_t, a_t) = Q(s_t, a_t) - V(s_t, a_t)$
        pred_advs = experience.returns[:-1] - experience.pred_state_values[:-1]
        pred_advs = (pred_advs - pred_advs.mean()) / (pred_advs.std() + self.epsilon)

        # Update n_epoch times
        action_loss_sum = 0.0; value_loss_sum = 0.0; action_distrib_entropy_sum = 0.0; loss_sum = 0.0
        for epoch_idx in range(self.n_epoch):
            sample_gen = experience.feed_forward_generator(pred_advs, self.n_minibatch)

            for samples in sample_gen:
                def closure():
                    _observs, _actions, _action_log_probs, _returns, _pred_advs = samples
                    action_log_probs, action_distrib_entropy, pred_state_values = self.actor_critic_net.evaluate_actions(_observs, _actions)

                    ratio = torch.exp(action_log_probs - _action_log_probs) # $\pi_{\theta}(a_t, s_t) / \pi{\theta_{old}}(a_t, s_t)$
                    surr1 = ratio * _pred_advs
                    surr2 = torch.clamp(ratio, (1.0 - self.clip_eps), (1.0 + self.clip_eps)) * _pred_advs

                    action_loss = - torch.min(surr1, surr2).mean()
                    value_loss = fn.mse_loss(pred_state_values, _returns) # other args: size_average=True, reduce=True
                    loss = action_loss + value_loss

                    self.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic_net.parameters(), self.max_grad_norm)

                    assert not(torch.isnan(loss))
                    nonlocal loss_sum, action_loss_sum, value_loss_sum, action_distrib_entropy_sum
                    loss_sum += loss.item()
                    action_loss_sum += action_loss.item()
                    value_loss_sum += value_loss.item()
                    action_distrib_entropy_sum += action_distrib_entropy.item()
                    return loss

                # Step the optim
                self.optim.step(closure)

        # Summarize losses
        # Note: nupdate below may not be equal to #iteration in the loop above since
        # in sampler generator, we set drop_last=False,
        # this also implies: do not use mean(), eg action_loss_array.mean()
        nupdate = self.n_epoch * self.n_minibatch
        loss = loss_sum / nupdate
        action_loss = action_loss_sum / nupdate
        value_loss = value_loss_sum / nupdate
        action_distrib_entropy = action_distrib_entropy_sum / nupdate

        return loss, value_loss, action_loss, action_distrib_entropy
