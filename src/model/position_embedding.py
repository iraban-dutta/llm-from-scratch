import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
from .llm_config import LLMConfig



class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.ctx_len=config.ctx_len
        self.d_model=config.d_model
        self.PE = self._get_positional_encoding()

    def _get_positional_encoding(self) -> torch.Tensor:
        # Creates an array PE of shape (ctx_len, d_model)
        # PE(pos, 2i) = sin(pos/10000**(2i/d_model))
        # PE(pos, 2i+1) = cos(pos/10000**((2*i+1)/d_model))

        PE = torch.zeros(self.ctx_len, self.d_model)
        for pos in range(self.ctx_len):
            for i in range(0, self.d_model//2):
                PE[pos, 2*i]   = math.sin(pos/(10000**((2*i)/self.d_model)))
                PE[pos, 2*i+1] = math.cos(pos/(10000**((2*i)/self.d_model)))

        return PE

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x.shape = (B, T, d_model)
        B, T, d_model = x.shape

        # Move PE to same device as x
        self.PE = self.PE.to(x.device)

        # We slice the PE tensor to match the #seqeunces in x
        # this is because the sequence length in x can vary from [1, ctx_len]
        return x + self.PE[:T, :]

    def visualize_PE(self) -> None:
        plt.figure(figsize=(12, 6))

        plt.imshow(
            self.PE,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            interpolation="nearest"
        )

        plt.xlabel("Embedding Dimension")
        plt.ylabel("Position")
        plt.colorbar()
        plt.tight_layout()
        plt.show()


class LearnedPostionalEmbedding(nn.Module):
    
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.PE = nn.Embedding(config.ctx_len, config.d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x.shape = (B, T, d_model)
        B, T, d_model = x.shape

        # Move PE to same device as x
        self.PE = self.PE.to(x.device)
        
        return x + self.PE(torch.arange(T))



def build_position_embedding(config:LLMConfig) -> nn.Module:

    match config.position_embedding:

        case "sinusoidal":
            return SinusoidalPositionalEncoding(config)
        case "learned":
            return LearnedPostionalEmbedding(config)
        case "identity":
            return nn.Identity()



if __name__=='__main__':

    # Test SinusoidalPostionalEncoding()
    ctx_len=50
    d_model=32
    config = LLMConfig(
        ctx_len=ctx_len, 
        d_model=d_model
    )

    pos_enc = SinusoidalPositionalEncoding(config)
    print(pos_enc)

    x = torch.randn(2, ctx_len, d_model)
    print(x.shape)
    x = pos_enc(x)
    print(x.shape)
    pos_enc.visualize_PE()
