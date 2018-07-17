import torch

class ExperienceBuffer():
    def __init__(self, nstep_per_update, nprocess, observ_dim, action_dim):
        self.rewards = torch.zeros(nstep_per_update, nprocess, 1)
        self.action_log_probs = torch.zeros(nstep_per_update, nprocess, 1)
        self.actions = torch.zeros(nstep_per_update, nprocess, action_dim)

        # Below, +1 to store next observ, returns, pred_values
        self.observs = torch.zeros(nstep_per_update + 1, nprocess, observ_dim)
        self.returns = torch.zeros(nstep_per_update + 1, nprocess, 1)
        self.pred_values = torch.zeros(nstep_per_update + 1, nprocess, 1)

        self.step_idx = 0
        self.nstep_per_update = nstep_per_update




