import torch
import torch.nn as nn
from .llm_config import LLMConfig



class LayerNorm(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(config.d_model))
        self.beta=nn.Parameter(torch.zeros(config.d_model)) if config.bias else None
        self.eps=config.eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x.shape = (B, T, d_model)

        l_mean = x.mean(dim=-1, keepdim=True)               # l_mean.shape = (B, T, 1)
        l_var = x.var(dim=-1, keepdim=True, unbiased=False) # l_var.shape  = (B, T, 1)

        x = (x - l_mean)/((l_var+self.eps)**0.5)
        x = x * self.gamma

        if self.beta is not None:
            x = x + self.beta

        return x


if __name__=='__main__':

    d_model=32
    eps=1e-5
    normalization='layernorm'

    config = LLMConfig(
        d_model=d_model,
        eps=eps, 
        bias=True,
        normalization=normalization, 
    )
    ln = LayerNorm(config)
    # print(list(ln.parameters()))
    for k, v in ln.state_dict().items():
        print(k, v.shape)

    x = torch.randn(2, 20, d_model)
    print(x.shape)
    x=ln(x)
    print(x.shape)

    

