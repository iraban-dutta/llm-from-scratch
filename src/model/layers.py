import torch
import torch.nn as nn
from .llm_config import LLMConfig
from .attention import build_attention
from .normalization import LayerNorm
from .moe import MLP, MoE
from src.inference.cache import KVCache, MHLACache


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
        self.mlp=MLP(config) if not config.use_moe else MoE(config)
        
    def forward(self, x:torch.Tensor, cache:KVCache|MHLACache|None=None) -> torch.Tensor:
        # x.shape = (B, T, d_model)

        # PreNorm Architecture with a clean residual stream
        x = x + self.attn(self.ln_attn(x), cache)
        x = x + self.mlp(self.ln_mlp(x))

        return x



if __name__=='__main__':

    d_model = 64

    # # Decoder with MLP config
    # config = LLMConfig(
    #     d_model=d_model, 
    #     ff_ratio=4,
    #     dropout=0.0,
    #     n_heads=4, 
    #     use_flash=False, 
    #     attn_debug=False,
    #     use_moe=False
    # )

    # Decoder with MoE config
    config = LLMConfig(
        d_model=d_model, 
        ff_ratio=4,
        dropout=0.0,
        n_heads=4, 
        use_flash=False, 
        attn_debug=False,
        use_moe=True,
        n_experts=3,
        topk=2,
        capcity_factor = 1.2
    )

    decoder = Decoder(config)
    for k, v in decoder.state_dict().items():
        print(k, v.shape)


    x = torch.randn(2, 20, d_model)
    print(x.shape)
    x = decoder(x)
    print(x.shape)
