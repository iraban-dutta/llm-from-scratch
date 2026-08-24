import torch
import torch.nn as nn
from .llm_config import LLMConfig
from .attention import build_attention
from .normalization import LayerNorm
from src.inference.kv_cache import KVCache


class MLP(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.proj_in=nn.Linear(in_features=config.d_model, out_features=config.ff_ratio*config.d_model, bias=config.bias)
        self.gelu=nn.GELU(approximate='tanh')
        self.proj_out=nn.Linear(in_features=config.ff_ratio*config.d_model, out_features=config.d_model, bias=config.bias)
        self.proj_out.RESIDUAL_PATH_SCALE_INIT=1

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.gelu(x)
        x = self.proj_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()
        # LayerNorm for Attention Block
        self.ln_attn=LayerNorm(config)
        # Attention Block
        self.attn=build_attention(config)
        # LayerNorm for MLP Block
        self.ln_mlp=LayerNorm(config)
        # MLP Block
        self.mlp=MLP(config)
        



    def forward(self, x:torch.Tensor, kv_cache:KVCache|None=None) -> torch.Tensor:
        # x.shape = (B, T, d_model)

        # PreNorm Architecture with a clean residual stream
        x = x + self.attn(self.ln_attn(x), kv_cache)
        x = x + self.mlp(self.ln_mlp(x))

        return x



if __name__=='__main__':

    d_model=64
    ff_ratio=4
    dropout=0.0
    n_heads=4
    use_flash=False
    attn_debug=False

    config = LLMConfig(
        d_model=d_model, 
        ff_ratio=ff_ratio,
        dropout=dropout,
        n_heads=n_heads, 
        use_flash=use_flash, 
        attn_debug=attn_debug
    )

    decoder = Decoder(config)
    for k, v in decoder.state_dict().items():
        print(k, v.shape)


    x = torch.randn(2, 20, d_model)
    print(x.shape)
    x = decoder(x)
    print(x.shape)
