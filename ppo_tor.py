import torch
import torch.nn.functional as fn

class VanillaPPO():
    def __init__(self, actor_critic_net, clip_eps, max_grad_norm, lr, nepoch, nminibatch, eps):
        self.actor_critic_net = actor_critic_net
        self.clip_eps = clip_eps
        self.nepoch = nepoch
        self.nminibatch = nminibatch
        self.max_grad_norm = max_grad_norm
        self.optim = torch.optim.Adam(actor_critic_net.parameters(), lr=lr, eps=eps)

    def update(self, rollouts, eps=1e-5):
        # Compute advantages: $A(s_t, a_t) = Q(s_t, a_t) - V(s_t, a_t)$
        pred_advs = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        pred_advs = (pred_advs - pred_advs.mean()) / (pred_advs.std() + eps)

        # Update in multiple epoches
        action_loss = []; value_loss = []; action_distrib_entropy = []

        for epoch_idx in range(self.nepoch):
            sample_gen = rollouts.feed_forward_generator(pred_advs, self.nminibatch)

            for samples in sample_gen:
                _observs, _actions, _action_log_probs, _returns, _pred_advs, _masks = samples
                pred_state_values, action_log_probs, _action_distrib_entropy = self.actor_critic_net.evaluate_actions(_observs, _actions)

                ratio = torch.exp(action_log_probs - _action_log_probs)
                surr1 = ratio * _pred_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * _pred_advs

                _action_loss = - torch.min(surr1, surr2).mean()
                _value_loss = fn.mse_loss(_returns, pred_state_values)
                loss = _action_loss + _value_loss

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic_net.parameters(), self.max_grad_norm)
                self.optim.step()

                action_loss.append(_action_loss)
                value_loss.append(_value_loss)
                action_distrib_entropy.append(_action_distrib_entropy)

        # Take mean of losses
        action_loss = torch.tensor(action_loss).mean()
        action_distrib_entropy = torch.tensor(action_distrib_entropy).mean()
        value_loss = torch.tensor(value_loss).mean()

        return value_loss, action_loss, action_distrib_entropy


