import torch
import torch.nn as nn

from distributions_tor import GaussianDistributionNetwork
from utils_tor import init_param_openaibaselines

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, actor_output_dim, critic_output_dim):
        super(ActorCriticNetwork, self).__init__()

        self.hidden_net = ActorCriticHiddenNetwork(input_dim)

        # Output networks
        self.actor_output_net = GaussianDistributionNetwork(self.hidden_net.hidden_dim, output_dim)
        self.critic_output_net = init_param_openaibaselines(nn.Linear(self.hidden_net.hidden_dim, critic_output_dim))

    def forward(self, inputs, states, masks):
        hidden_actor = self.hidden_net.actor_hidden_net(inputs)
        hidden_critic = self.hidden_net.critic_hidden_net(inputs)

        return self.critic_output_net(hidden_critic), hidden_actor, states

class ActorCriticHiddenNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ActorCriticHiddenNetwork, self).__init__()
        hidden_dim = 64

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

        self.hidden_dim = hidden_dim

    def forward(self, inputs, states, masks):
        raise NotImplementedError




