import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from einops import einsum, rearrange, reduce, repeat
from .llm_config import LLMConfig
from .position_embedding import RoPE
from src.inference.kv_cache import KVCache


class GroupedQueryAttention(nn.Module):
    
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.d_model=config.d_model
        self.n_heads=config.n_heads
        self.n_groups=config.n_groups
        self.q_dim = config.d_model//config.n_heads
        # QKV projection layer
        self.proj_qkv = nn.Linear(config.d_model, config.d_model + 2*self.n_groups*self.q_dim)
        # Output projection layer
        self.proj_out = nn.Linear(config.d_model, config.d_model)
        self.proj_out.RESIDUAL_PATH_SCALE_INIT=1
        # Rotary Embedding
        self.rope = RoPE(head_dim=self.q_dim) if config.rotary_embedding else None
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


    def forward(self, x:torch.Tensor, kv_cache:KVCache|None=None) -> torch.Tensor:
        # x.shape = (B, T, d_model)
        B, T, _ = x.shape

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Get qkv projections in concated form
        # -------------------------------- XX -------------------------------- XX --------------------------------
        qkv = self.proj_qkv(x)

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Split q, k, v
        # -------------------------------- XX -------------------------------- XX --------------------------------
        q, k, v = torch.split(qkv, [self.d_model, self.n_groups*self.q_dim, self.n_groups*self.q_dim], dim=-1)

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Reshape q, k, v by accounting for n_heads, n_groups
        # -------------------------------- XX -------------------------------- XX --------------------------------
        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k, "b s (g d) -> b g s d", g=self.n_groups)
        v = rearrange(v, "b s (g d) -> b g s d", g=self.n_groups)

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Apply RoPE
        # -------------------------------- XX -------------------------------- XX --------------------------------
        use_kv_cache = kv_cache is not None
        if self.rope is not None:
            if use_kv_cache:
                # This path used during Inference (Prefill + Decode) with KV Cache enaabled
                # Handles sequences during inference time whihc are longer than ctx_len supported
                # Slides context of the last ctx_len tokens
                # Bake absolute postion into KV cache by rotating by m(pos of current token in seq)
                # This does not matter since during attention calculation, we only get relative position
                q = self.rope.apply_rope(x=q, seq_offset=kv_cache.ntokens_processed)
                k = self.rope.apply_rope(x=k, seq_offset=kv_cache.ntokens_processed)
            else:
                # This path used during Train + Naive Inference
                q = self.rope.apply_rope(x=q)
                k = self.rope.apply_rope(x=k)


        # -------------------------------- XX -------------------------------- XX --------------------------------
        # KV Caching: Activated during Inference if kv_cache is an object of correct class
        # -------------------------------- XX -------------------------------- XX --------------------------------
        if use_kv_cache:

            # Resolve Prefill/Decode stage (this controls behaviour for causal masking)
            is_decode = kv_cache.k_cache is not None

            # Update the KV Cache 
            kv_cache.update_cache(k, v)

            # Prefill Stage: Update the k and v vectors corresponding to all q vectors from prompt tokens, then fetch entire k,v (overrwriting)
            # Decode Stage: Update the k and v vector corresponding to current q (current token), then fetch entire k,v (past + current token)
            # FIX: We are using a KV cache whose size gets allocated at the start of inference, hence very important to slice it upto current token position
            k = kv_cache.k_cache[:, :, :kv_cache.curr_idx] 
            v = kv_cache.v_cache[:, :, :kv_cache.curr_idx]


        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Repeat k v to be shared across q heads 
        # -------------------------------- XX -------------------------------- XX --------------------------------

        # Ideally NOT correct: eg: G0 G1 G0 G1 (KV groups become interleaved)
        # k = repeat(k, "b g s d -> b (n g) s d", n=self.n_heads//self.n_groups)
        # v = repeat(v, "b g s d -> b (n g) s d", n=self.n_heads//self.n_groups)

        # Correct: eg: G0 G0 G1 G1
        k = repeat(k, "b g s d -> b (g n) s d", n=self.n_heads//self.n_groups)
        v = repeat(v, "b g s d -> b (g n) s d", n=self.n_heads//self.n_groups)


        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Flag to control causal masking behaviour
        # -------------------------------- XX -------------------------------- XX --------------------------------
        is_causal_flag=True             # Usual Flow (Without KVCache/With KV:Train_Infer-Prefill)
        if use_kv_cache and is_decode:
            is_causal_flag=False        # Overrides flag to false only when KVCaching is used + Infer-Decode


        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Attention
        # -------------------------------- XX -------------------------------- XX --------------------------------
        if self.use_flash:
            # Implement Flash Attention 
            y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=self.dropout_p if self.training else 0, 
                    is_causal=is_causal_flag
                )
        else:
            # Implement Manual Attention 
            # Attention_p1: Dot Product (Q @ K.T)
            attn_scores = einsum(q, k, "b h i d, b h j d -> b h i j")

            # Attention_p2: Scaling
            head_dim = self.d_model//self.n_heads
            attn_scores *= (1/(head_dim)**0.5)

            # Attention_p3: Causal Mask
            if is_causal_flag:
                # Way1
                mask = torch.triu(torch.full_like(attn_scores, fill_value=-torch.inf, device=x.device), diagonal=1)
                attn_scores += mask

                # # Way2
                # mask = torch.tril(torch.ones(T, T, device=x.device))==0
                # attn_scores = attn_scores.masked_fill(mask, value=-torch.inf)

            # Attention_p4: Softmax
            attn_scores = F.softmax(attn_scores, dim=-1)
            if self.attn_debug:
                self.attn_debug_probs = attn_scores.detach()
            attn_scores = self.dropout_attn(attn_scores)

            # Attention_p5: Dot Product (A @ V)
            y = einsum(attn_scores, v, "b h i j, b h j d -> b h i d")

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Reshape post attention output to original shape of input
        # -------------------------------- XX -------------------------------- XX --------------------------------
        y = rearrange(y, "b h s d -> b s (h d)")

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Get op projection
        # -------------------------------- XX -------------------------------- XX --------------------------------
        y = self.dropout_out(self.proj_out(y))


        return y




class MultiHeadAttention(nn.Module):

    def __init__(self, config:LLMConfig):
        super().__init__()
        self.n_heads=config.n_heads
        self.d_model=config.d_model
        # QKV projection layer
        self.proj_qkv=nn.Linear(config.d_model, 3*config.d_model)
        # Output projection layer
        self.proj_out=nn.Linear(config.d_model, config.d_model)
        self.proj_out.RESIDUAL_PATH_SCALE_INIT=1
        # Rotary Embedding
        self.rope = RoPE(head_dim=self.q_dim) if config.rotary_embedding else None
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



    def forward(self, x:torch.Tensor, kv_cache:KVCache|None=None) -> torch.Tensor:
        # x.shape = (B, T, d_model)
        B, T, d_model = x.shape

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Get qkv projections in concated form
        # -------------------------------- XX -------------------------------- XX --------------------------------
        qkv = self.proj_qkv(x)

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Split q, k, v
        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Dirty way
        # q = qkv[:, :, 0*d_model:1*d_model]
        # k = qkv[:, :, 1*d_model:2*d_model]
        # v = qkv[:, :, 2*d_model:3*d_model]

        # Clean way
        q, k, v = torch.split(qkv, self.d_model, dim=-1)


        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Reshape q, k, v by accounting for n_heads
        # -------------------------------- XX -------------------------------- XX --------------------------------
        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Apply RoPE
        # -------------------------------- XX -------------------------------- XX --------------------------------
        use_kv_cache = kv_cache is not None
        if self.rope is not None:
            if use_kv_cache:
                # This path used during Inference (Prefill + Decode) with KV Cache enaabled
                # Handles sequences during inference time whihc are longer than ctx_len supported
                # Slides context of the last ctx_len tokens
                # Bake absolute postion into KV cache by rotating by m(pos of current token in seq)
                # This does not matter since during attention calculation, we only get relative position
                q = self.rope.apply_rope(x=q, seq_offset=kv_cache.ntokens_processed)
                k = self.rope.apply_rope(x=k, seq_offset=kv_cache.ntokens_processed)
            else:
                # This path used during Train + Naive Inference
                q = self.rope.apply_rope(x=q)
                k = self.rope.apply_rope(x=k)


        # -------------------------------- XX -------------------------------- XX --------------------------------
        # KV Caching: Activated during Inference if kv_cache is an object of correct class
        # -------------------------------- XX -------------------------------- XX --------------------------------
        if use_kv_cache:

            # Resolve Prefill/Decode stage (this controls behaviour for causal masking)
            is_decode = kv_cache.k_cache is not None

            # Update the KV Cache 
            kv_cache.update_cache(k, v)

            # Prefill Stage: Update the k and v vectors corresponding to all q vectors from prompt tokens, then fetch entire k,v (overrwriting)
            # Decode Stage: Update the k and v vector corresponding to current q (current token), then fetch entire k,v (past + current token)
            # FIX: We are using a KV cache whose size gets allocated at the start of inference, hence very important to slice it upto current token position
            k = kv_cache.k_cache[:, :, :kv_cache.curr_idx] 
            v = kv_cache.v_cache[:, :, :kv_cache.curr_idx]


        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Flag to control causal masking behaviour
        # -------------------------------- XX -------------------------------- XX --------------------------------
        is_causal_flag=True             # Usual Flow (Without KVCache/With KV:Train_Infer-Prefill)
        if use_kv_cache and is_decode:
            is_causal_flag=False        # Overrides flag to false only when KVCaching is used + Infer-Decode

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Attention
        # -------------------------------- XX -------------------------------- XX --------------------------------
        if self.use_flash:
            # Implement Flash Attention 
            y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=self.dropout_p if self.training else 0, 
                    is_causal=is_causal_flag
                )
        else:
            # Implement Manual Attention 
            # Attention_p1: Dot Product (Q @ K.T)
            attn_scores = einsum(q, k, "b h i d, b h j d -> b h i j")

            # Attention_p2: Scaling
            head_dim = self.d_model//self.n_heads
            attn_scores *= (1/(head_dim)**0.5)

            # Attention_p3: Causal Mask
            if is_causal_flag:
                mask = torch.tril(torch.ones(T, T, device=x.device))==0
                attn_scores = attn_scores.masked_fill(mask, value=-torch.inf)

            # Attention_p4: Softmax
            attn_scores = F.softmax(attn_scores, dim=-1)
            if self.attn_debug:
                self.attn_debug_probs = attn_scores.detach()
            attn_scores = self.dropout_attn(attn_scores)

            # Attention_p5: Dot Product (A @ V)
            y = einsum(attn_scores, v, "b h i j, b h j d -> b h i d")

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Reshape post attention output to original shape of input
        # -------------------------------- XX -------------------------------- XX --------------------------------
        y = rearrange(y, "b h s d -> b s (h d)")

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Get op projection
        # -------------------------------- XX -------------------------------- XX --------------------------------
        y = self.dropout_out(self.proj_out(y))

        return y


def build_attention(config:LLMConfig) -> nn.Module:

    match config.attention:
        case "mha":
            return MultiHeadAttention(config)
        case "gqa":
            return GroupedQueryAttention(config)
        case "mhla":
            raise Exception (
                f'{config.attention} currently not supported!'
            )


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

    
    d_model=16
    n_heads=4
    n_groups=2
    use_flash=False
    attn_debug=False
    attention='gqa'

    config = LLMConfig(
        d_model=d_model, 
        attention=attention,
        n_heads=n_heads, 
        n_groups=n_groups,
        use_flash=use_flash, 
        attn_debug=attn_debug,
        rotary_embedding=True
    )
    mha = MultiHeadAttention(config)
    gqa = GroupedQueryAttention(config)
    # print(mha.training)
    # mha.eval()
    # print(mha.training)

    x = torch.randn(2, 20, d_model)
    print(x.shape)
    # x = mha(x)
    x = gqa(x)
    print(x.shape)

    if mha.attn_debug_probs is not None:
        print(mha.attn_debug_probs.shape)
        visualize_attnscores(mha.attn_debug_probs, mha.n_heads)

    if gqa.attn_debug_probs is not None:
        print(gqa.attn_debug_probs.shape)
        visualize_attnscores(gqa.attn_debug_probs, gqa.n_heads)

