import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from einops import einsum, rearrange, reduce, repeat
from .llm_config import LLMConfig


class MultiHeadAttention(nn.Module):

    def __init__(self, config:LLMConfig):
        super().__init__()
        self.n_heads=config.n_heads
        # QKV projection layer
        self.proj_attn=nn.Linear(config.d_model, 3*config.d_model)
        # Output projection layer
        self.proj_out=nn.Linear(config.d_model, config.d_model)
        # Dropouts
        self.dropout_attn = nn.Dropout(config.dropout)
        self.dropout_out = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout
        # Flash Attn
        self.use_flash = config.use_flash
        if self.use_flash:
            if hasattr(F, 'scaled_dot_product_attention'):
                print("use_flash=True - To use PyTorch's Flash Attention")
            else:
                self.use_flash=False
                print("use_flash=True - PyTorch's Flash Attention needs version >= 2.0 - Fallback to Manual Attention")
        # Debug Mode
        self.attn_debug = config.attn_debug
        self.attn_debug_probs = None



    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x.shape = (B, T, d_model)
        B, T, d_model = x.shape
        # Get qkv projections in concated form
        qkv = self.proj_attn(x)

        # Break into q, k, v: Dirty way
        # q = qkv[:, :, 0*d_model:1*d_model]
        # k = qkv[:, :, 1*d_model:2*d_model]
        # v = qkv[:, :, 2*d_model:3*d_model]
        # Break into q, k, v: Clean way
        q, k, v = torch.split(qkv, d_model, dim=-1)

        # Reshape q, k, v by accounting for n_heads
        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)

        if self.use_flash:
            # Implement Flash Attention 
            y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=self.dropout_p if self.training else 0, 
                    is_causal=True
                )
        else:
            # Implement Manual Attention 
            # Attention_p1: Dot Product (Q @ K.T)
            attn_scores = einsum(q, k, "b h i d, b h j d -> b h i j")
            # Attention_p2: Scaling
            head_dim = d_model//self.n_heads
            attn_scores *= (1/(head_dim)**0.5)
            # Attention_p3: Causal Mask
            mask = torch.tril(torch.ones(T, T, device=x.device))==0
            attn_scores = attn_scores.masked_fill(mask, value=-torch.inf)
            # Attention_p4: Softmax
            attn_scores = F.softmax(attn_scores, dim=-1)
            if self.attn_debug:
                self.attn_debug_probs = attn_scores.detach()
            attn_scores = self.dropout_attn(attn_scores)
            # Attention_p5: Dot Product (A @ V)
            y = einsum(attn_scores, v, "b h i j, b h j d -> b h i d")

        # Reshape post attention output to original shape of input
        y = rearrange(y, "b h s d -> b s (h d)")

        # Get op projection
        y = self.dropout_out(self.proj_out(y))

        return y


class MultiQueryAttention(nn.Module):
    pass

class GroupedQueryAttention(nn.Module):
    pass







def visualize_attnscores(t:torch.Tensor, n_heads:int, head:int=0) -> None:
    # t.shape = (B, H, T, T)
    if t is None:
        print('Run Forward Pass first to visualize Attention Matrix')
    else:
        # Only Visualize the 1st sample in the batch
        print(type(t))
        attn_scores = t[0]

        if head>=0 and head<=(n_heads-1):
            # Visualize a particular head

            plt.figure(figsize=(5, 5))
            plt.imshow(
                attn_scores[head],
                aspect="auto",
                cmap="Blues",
                vmin=0,
                vmax=1,
                interpolation="nearest"
            )
            plt.title(f"Head {head}")
            plt.xlabel("Key position")
            plt.ylabel("Query position")
            plt.tight_layout()
            plt.show()

        elif head==-1:
            # Visualize all heads
            n_cols = math.ceil(math.sqrt(n_heads))
            n_rows = math.ceil(n_heads / n_cols)

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(3 * n_cols, 3 * n_rows)
            )
            axes = axes.flatten()

            for i in range(n_heads):
                axes[i].imshow(
                    attn_scores[i],
                    aspect="auto",
                    cmap="Blues",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest"
                )
                axes[i].set_title(f"Head {i}")
                axes[i].set_xlabel("Key position")
                axes[i].set_ylabel("Query position")
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError(
                f"Head to visualize needs to be within [0, n_heads-1] or -1(for all heads)."
            )

    
if __name__=='__main__':

    
    d_model=36
    n_heads=6
    use_flash=False
    attn_debug=True

    config = LLMConfig(
        d_model=d_model, 
        n_heads=n_heads, 
        use_flash=use_flash, 
        attn_debug=attn_debug
    )
    mha = MultiHeadAttention(config)
    # print(mha.training)
    # mha.eval()
    # print(mha.training)

    x = torch.randn(2, 20, d_model)
    print(x.shape)
    x = mha(x)
    print(x.shape)

    if mha.attn_debug_probs is not None:
        print(mha.attn_debug_probs.shape)
        visualize_attnscores(mha.attn_debug_probs, mha.n_heads)

