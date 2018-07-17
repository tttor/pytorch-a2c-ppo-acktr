import torch

class ExperienceBuffer():
    def __init__(self, nstep_per_update, nprocess, observ_dim, action_dim):
        self.rewards = torch.zeros(nstep_per_update, nprocess, 1)
        self.action_log_probs = torch.zeros(nstep_per_update, nprocess, 1)
        self.actions = torch.zeros(nstep_per_update, nprocess, action_dim)

        # Below, +1 to store next observ, returns, pred_state_values
        self.observations = torch.zeros(nstep_per_update + 1, nprocess, observ_dim)
        self.returns = torch.zeros(nstep_per_update + 1, nprocess, 1)
        self.pred_state_values = torch.zeros(nstep_per_update + 1, nprocess, 1)

        self.step_idx = 0
        self.nstep_per_update = nstep_per_update

    def insert(self, action, action_log_prob, state_value, reward, next_observ):
        self.actions[self.step_idx].copy_(action)
        self.action_log_probs[self.step_idx].copy_(action_log_prob)
        self.pred_state_values[self.step_idx].copy_(state_value)
        self.rewards[self.step_idx].copy_(reward)
        self.observations[self.step_idx+1].copy_(next_observ)

        self.step_idx += 1
        self.step_idx %= self.nstep_per_update
