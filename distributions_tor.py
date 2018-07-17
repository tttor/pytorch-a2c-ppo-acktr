import torch
import torch.nn as nn

from utils_tor import init_param_openaibaselines

class GaussianDistributionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GaussianDistributionNetwork, self).__init__()
        self.mean_net = init_param_openaibaselines(nn.Linear(input_dim, output_dim))
        self.logstd_net = MeanBiasNetwork(output_dim)

    def forward(self, x):
        mean = self.mean_net(x)
        logstd = self.logstd_net(torch.zeros_like(mean))
        return torch.distributions.Normal(loc=mean, scale=torch.exp(logstd))

class MeanBiasNetwork(nn.Module):
    def __init__(self, output_dim):
        super(MeanBiasNetwork, self).__init__()
        init_param = torch.zeros(output_dim)
        self._param = nn.Parameter(init_param.unsqueeze(dim=1))

    def forward(self, x):
        assert x.dim()==2
        bias = self._param.t().view(1, -1)
        return x + bias
