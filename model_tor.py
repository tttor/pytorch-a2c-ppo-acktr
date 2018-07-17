import torch
import torch.nn as nn

from distributions_tor import GaussianDistributionNetwork
from utils_tor import init_param_openaibaselines

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, actor_output_dim, critic_output_dim):
        super(ActorCriticNetwork, self).__init__()

        hidden_dim = 64
        self.hidden_net = ActorCriticHiddenNetwork(input_dim, hidden_dim, critic_output_dim)

        # Output networks
        self.actor_output_net = GaussianDistributionNetwork(hidden_dim, actor_output_dim)
        # self.critic_output_net = init_param_openaibaselines(nn.Linear(hidden_dim, critic_output_dim)) # TODO: uncomment me, rm the one in hiddenNet
        self.state_size = critic_output_dim # TODO: make naming more descriptive

    def act(self, observ):
        state_value, meta_action = self._forward(observ)

        action_distrib = self.actor_output_net(meta_action)
        action = action_distrib.sample()
        action_log_prob = action_distrib.log_prob(action).sum(dim=-1, keepdim=True)

        return action, action_log_prob, state_value

    def predict_state_value(self, observ):
        state_value, _ = self._forward(observ)
        return state_value

    def evaluate_actions(self, observ, action):
        value, meta_action = self._forward(observ)

        action_distrib = self.actor_output_net(meta_action)
        action_log_prob = action_distrib.log_prob(action).sum(dim=-1, keepdim=True)
        action_distrib_entropy = action_distrib.entropy().sum(dim=-1, keepdim=False).mean()

        return value, action_log_prob, action_distrib_entropy

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def _forward(self, observ):
        hidden_actor = self.hidden_net.actor_hidden_net(observ)
        hidden_critic = self.hidden_net.critic_hidden_net(observ)
        return self.hidden_net.critic_output_net(hidden_critic), hidden_actor

class ActorCriticHiddenNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, critic_output_dim):
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

        self.critic_output_net = init_param_openaibaselines(nn.Linear(hidden_dim, critic_output_dim)) # TODO: move to ActorCriticNet()

    def forward(self, observ):
        raise NotImplementedError