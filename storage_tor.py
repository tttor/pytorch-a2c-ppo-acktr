import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class ExperienceBuffer():
    def __init__(self, nstep_per_update, nprocess, observ_dim, action_dim):
        self.rewards = torch.zeros(nstep_per_update, nprocess, 1)
        self.action_log_probs = torch.zeros(nstep_per_update, nprocess, 1)
        self.actions = torch.zeros(nstep_per_update, nprocess, action_dim)

        # Below +1 is to store next observ, next returns, next pred_state_values
        self.observations = torch.zeros(nstep_per_update + 1, nprocess, observ_dim)
        self.returns = torch.zeros(nstep_per_update + 1, nprocess, 1)
        self.pred_state_values = torch.zeros(nstep_per_update + 1, nprocess, 1)
        self.masks = torch.ones(nstep_per_update + 1, nprocess, 1)

        self.step_idx = 0
        self.nstep_per_update = nstep_per_update
        self.nprocess = nprocess
        self.observ_dim = observ_dim
        self.action_dim = action_dim

    def insert(self, action, action_log_prob, state_value, reward, next_observ, next_mask):
        self.actions[self.step_idx].copy_(action)
        self.action_log_probs[self.step_idx].copy_(action_log_prob)
        self.pred_state_values[self.step_idx].copy_(state_value)
        self.rewards[self.step_idx].copy_(reward)
        self.observations[self.step_idx+1].copy_(next_observ)
        self.masks[self.step_idx+1].copy_(next_mask)

        self.step_idx += 1
        self.step_idx %= self.nstep_per_update

    def compute_returns(self, pred_next_state_value, gamma):
        self.returns[-1] = pred_next_state_value # TODO: why set this?
        for i in reversed(range(self.rewards.shape[0])): # i: reversed step_idx
            self.returns[i] = self.rewards[i] + (gamma * self.returns[i+1] * self.masks[i+1])

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def feed_forward_generator(self, _advantages, nminibatch):
        batch_size = self.nstep_per_update * self.nprocess
        assert batch_size >= nminibatch

        minibatch_size = batch_size // nminibatch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), minibatch_size, drop_last=False)

        for idxs in sampler:
            observs = self.observations[:-1].view(-1, self.observ_dim)[idxs]
            returns = self.returns[:-1].view(-1, 1)[idxs]
            masks = self.masks[:-1].view(-1, 1)[idxs]

            actions = self.actions.view(-1, self.action_dim)[idxs]
            action_log_probs = self.action_log_probs.view(-1, 1)[idxs]
            advantages = _advantages.view(-1, 1)[idxs]

            yield observs, actions, action_log_probs, returns, advantages, masks
