import torch

def init_param_openaibaselines(module):
    # https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
    def _init_normc(weight, gain):
        weight.normal_(mean=0.0, std=1.0)
        weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

    _init_normc(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module
