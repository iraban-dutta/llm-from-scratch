import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt
from .llm_config import LLMConfig
from einops import einsum, rearrange, reduce, repeat


class RoPE:
    def __init__(self, head_dim:int, base:int=10000):

        # Validate that head_dim should be an even number
        assert head_dim % 2 == 0, (
            f'Head dimension should be even, got {head_dim}'
        )

        self.head_dim = head_dim
        self.base = base

    def _get_m_thetas(self, x:torch.Tensor, seq_len:int, seq_offset:int) -> torch.Tensor:

        # Step1: Build thetas | Shape: (d/2, )
        # 1/theta_i = 10000**(2(i-1)/d) for all i in [1, 2, 3, ...d/2]

        theta_denom = self.base**(torch.arange(0, self.head_dim, 2, device=x.device)/self.head_dim)
        theta = 1/theta_denom

        # Step2: Build ms  | Shape: (m, )
        m = torch.arange(seq_offset+1, seq_offset+1+seq_len, device=x.device)

        # Step3: Take outer product | Shape: (m, d/2)
        # (can use torch.outer as well) 
        m_thetas = einsum(m, theta, "t, d -> t d")

        # Replicate m_thetas across dim=-1 | Shape: (m, d)
        m_thetas = repeat(m_thetas, "t d -> t (d n)", n=2)

        return m_thetas


    def _get_x_flipped(self, x:torch.Tensor) -> torch.Tensor:

        # Step1: Reshape the x to the below form (x -> x_flipped)
        # Basic flipping idea
        # x =         [[x11 x12 x13 x14], 
        #              [x21 x22 x23 x24]]
        # x_flipped = [[-x12 x11 -x14 x13], 
        #              [-x22 x21 -x24 x23]]

        # Actual x will be in higher dim (since it will have B:batch and H:head dimension)
        # x.shape: (B, H, m, d)
        B, H, m, _ = x.shape

        # 1.1: Copy and reshape
        x_copy = x.view(B, H, m, -1, 2)     # Shape: (B, H, m, d/2, 2)

        # 1.2: Pluck out the even and odd dims
        x_copy1 = x_copy[:, :, :, :, 0]          # Shape: (B, H, m, d/2) | x_copy1=x_copy[..., 0] (Clean way to write)
        x_copy2 = x_copy[:, :, :, :, 1]          # Shape: (B, H, m, d/2) | x_copy2=x_copy[..., 1] (Clean way to write)

        # 1.3: Stack up the plucked out tensors in correct order with negative sign
        x_flipped = torch.stack(
            [-x_copy2, x_copy1], 
            dim=-1
        )                                        # Shape: (B, H, m, d/2, 2)

        # 1.4 # Reshape back to original shape as x
        x_flipped = x_flipped.view(B, H, m, -1)  # Shape: (B, H, m, d)

        return x_flipped


    def apply_rope(self, x:torch.Tensor, seq_offset:int=0) -> torch.Tensor:
        # x.shape: B, H, m, d
        _, _, m, _ = x.shape

        # 1: Get m_thetas 
        # m_thetas.shape : (m, d)
        m_thetas = self._get_m_thetas(x, seq_len=m, seq_offset=seq_offset)

        # 2: Create cos(m_thetas) and sin(m_thetas) 
        # Shape          : (m, d)
        m_theta_cos = torch.cos(m_thetas)
        m_theta_sin = torch.sin(m_thetas)

        # 3. Get x_flipped 
        # x.shape        : (B, H, m, d)
        # x_flipped.shape: (B, H, m, d)
        x_flipped = self._get_x_flipped(x)

        # 4: Final additions to create rotated vector: Element wise product (Implicit: Broadcasting happens here)
        # x_rot.shape    : (B, H, m, d)
        x_rot = (x*m_theta_cos) + (x_flipped*m_theta_sin)

        return x_rot



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

    def forward(self, x:torch.Tensor, position_offset:int=0) -> torch.Tensor:
        # x.shape = (B, T, d_model)
        B, T, d_model = x.shape

        # Move PE to same device as x
        self.PE = self.PE.to(x.device)

        # We slice the PE tensor to match the sequence length in x, ie the #tokens(T)
        # This is because the sequence length in x can vary from [1, ctx_len]
        # Tensor Addition has implicit broadcastiing in batch dimension
        return x + self.PE[position_offset:position_offset+T, :]

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

    def forward(self, x:torch.Tensor, position_offset:int=0) -> torch.Tensor:
        # x.shape = (B, T, d_model)
        B, T, d_model = x.shape

        # Move PE to same device as x
        self.PE = self.PE.to(x.device)
        
        return x + self.PE(torch.arange(position_offset, position_offset + T, device=x.device))


class ByPassPostionalEmbedding(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()

    def forward(self, x:torch.Tensor, position_offset:int=0) -> torch.Tensor:
        # x.shape = (B, T, d_model)
        return x


def build_position_embedding(config:LLMConfig) -> nn.Module:

    match config.position_embedding:

        case "sinusoidal":
            return SinusoidalPositionalEncoding(config)
        case "learned":
            return LearnedPostionalEmbedding(config)
        case "identity":
            return ByPassPostionalEmbedding(config)



if __name__=='__main__':

    # # Test SinusoidalPostionalEncoding()
    # ctx_len=50
    # d_model=32
    # config = LLMConfig(
    #     ctx_len=ctx_len, 
    #     d_model=d_model
    # )

    # pos_enc = SinusoidalPositionalEncoding(config)
    # print(pos_enc)

    # x = torch.randn(2, ctx_len, d_model)
    # print(x.shape)
    # x = pos_enc(x)
    # print(x.shape)
    # pos_enc.visualize_PE()


    # Test RoPE
    x = torch.randn(2, 4, 5, 6)
    print(x.shape)
    x_rot = RoPE(6).apply_rope(x)
    print(x_rot.shape)
