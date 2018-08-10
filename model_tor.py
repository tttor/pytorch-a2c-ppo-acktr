import torch
import torch.nn as nn

from distributions_tor import GaussianDistributionNetwork
from utils_tor import init_param_openaibaselines

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, actor_output_dim, critic_output_dim):
        super(ActorCriticNetwork, self).__init__()
        self.hidden_net = ActorCriticHiddenNetwork(input_dim, hidden_dim)
        self.actor_output_net = GaussianDistributionNetwork(hidden_dim, actor_output_dim)
        self.critic_output_net = init_param_openaibaselines(nn.Linear(hidden_dim, critic_output_dim))

    def act(self, observ):
        state_value, meta_action = self._forward(observ)
        action_distrib = self.actor_output_net(meta_action)
        action = action_distrib.sample()
        action_log_prob = action_distrib.log_prob(action).sum(dim=-1, keepdim=True)
        return action, action_log_prob, state_value

    def evaluate_actions(self, observ, action):
        state_value, meta_action = self._forward(observ)
        action_distrib = self.actor_output_net(meta_action)
        action_log_prob = action_distrib.log_prob(action).sum(dim=-1, keepdim=True)
        action_distrib_entropy = action_distrib.entropy().sum(dim=-1, keepdim=False).mean()
        return action_log_prob, action_distrib_entropy, state_value

    def predict_state_value(self, observ):
        hidden_critic = self.hidden_net.critic_hidden_net(observ)
        return self.critic_output_net(hidden_critic)

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def _forward(self, observ):
        meta_action = self.hidden_net.actor_hidden_net(observ)
        state_value = self.predict_state_value(observ)
        return (state_value, meta_action)

class ActorCriticHiddenNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ActorCriticHiddenNetwork, self).__init__()

        self.actor_hidden_net = nn.Sequential(
            init_param_openaibaselines(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            init_param_openaibaselines(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh()
        )

        self.critic_hidden_net = nn.Sequential(
            init_param_openaibaselines(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            init_param_openaibaselines(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh()
        )

    def forward(self, observ):
        raise NotImplementedError
