import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from einops import einsum, rearrange, reduce, repeat
from .llm_config import LLMConfig
from .position_embedding import RoPE
from src.inference.cache import KVCache, MHLACache
from typing import Tuple


def compute_attention(
        q:torch.Tensor, 
        k:torch.Tensor, 
        v:torch.Tensor, 
        use_flash:bool,
        dropout_p:float, 
        is_causal_flag:bool, 
        head_dim:int,
        attn_debug:bool,
        dropout_attn:nn.Module,
        device=torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor|None]:

    attn_debug_probs = None
    if use_flash:
        # PyTorch's Flash Attention - launches optimized kernels
        y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=dropout_p, 
                is_causal=is_causal_flag
            )
    else:
        # Implement Manual Attention 
        # Attention_p1: Dot Product (Q @ K.T)
        attn_scores = einsum(q, k, "b h i d, b h j d -> b h i j")

        # Attention_p2: Scaling
        attn_scores *= (1/(head_dim)**0.5)

        # Attention_p3: Causal Mask
        if is_causal_flag:
            # Way1
            mask = torch.triu(torch.full_like(attn_scores, fill_value=-torch.inf, device=device), diagonal=1)
            attn_scores += mask

            # # Way2
            # mask = torch.tril(torch.ones(T, T, device=x.device))==0
            # attn_scores = attn_scores.masked_fill(mask, value=-torch.inf)

        # Attention_p4: Softmax
        attn_scores = F.softmax(attn_scores, dim=-1)
        if attn_debug:
            attn_debug_probs = attn_scores.detach()
        attn_scores = dropout_attn(attn_scores)

        # Attention_p5: Dot Product (A @ V)
        y = einsum(attn_scores, v, "b h i j, b h j d -> b h i d")

    return y, attn_debug_probs


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()
        # Projection layers (Without RoPE): Dimensions
        self.d_model=config.d_model
        self.n_heads=config.n_heads
        self.d_latent1=config.d_latent1
        self.d_latent2=config.d_latent2
        self.d_head=(config.d_model//config.n_heads)
        # Projection layers (Without RoPE): Linear Layers
        self.proj_down_kv = nn.Linear(config.d_model, config.d_latent1)
        self.proj_up_k    = nn.Linear(config.d_latent1, config.n_heads*self.d_head, bias=False)
        self.proj_up_v    = nn.Linear(config.d_latent1, config.n_heads*self.d_head)
        self.proj_down_q  = nn.Linear(config.d_model, config.d_latent2)
        self.proj_up_q    = nn.Linear(config.d_latent2, config.n_heads*self.d_head, bias=False)
        # Rotary Embedding
        # Projection layers (With RoPE): Dimensions
        self.d_headR = config.d_headR if config.rotary_embedding else 0
        self.rope = RoPE(head_dim=config.d_headR) if config.rotary_embedding else None
        if self.rope is not None:
            # Projection layers (With RoPE): Linear Layers
            self.proj_kR = nn.Linear(config.d_model, config.d_headR)                  # kR is shared across all heads
            self.proj_qR = nn.Linear(config.d_latent2, config.n_heads*config.d_headR) # qR is unique for each head
        # Output projection layer
        self.proj_out = nn.Linear(config.d_model, config.d_model)
        self.proj_out.RESIDUAL_PATH_SCALE_INIT=1
        # Dropouts
        self.dropout_attn = nn.Dropout(config.dropout)
        self.dropout_out = nn.Dropout(config.dropout)
        # Debug Mode
        self.attn_debug = config.attn_debug
        self.attn_debug_probs = None


    def forward(self, x:torch.Tensor, mhla_cache:MHLACache|None=None) -> torch.tensor:

        # x.shape = (B, T, d_model)
        B, T, _ = x.shape

        use_mhla_cache = mhla_cache is not None

        # -------------------------------- XX -------------------------------- 
        # Get latent_Q, Q and reshape Q across heads
        # -------------------------------- XX -------------------------------- 
        latent_q  = self.proj_down_q(x)                              # shape: (B, (T/1), d_latent2)
        q = self.proj_up_q(latent_q)                                 # shape: (B, (T/1), H*d_head)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)     # shape: (B, H, (T/1), d_head)

        # -------------------------------- XX -------------------------------- 
        # Get latent_KV and Absorbed Q
        # -------------------------------- XX -------------------------------- 
        latent_kv = self.proj_down_kv(x)                                            # shape: (B, (T/1), d_latent1) 
        W_UK = rearrange(self.proj_up_k.weight, "(h d) l -> h d l", h=self.n_heads) # shape: (H*d_head, d_latent1) -> (H, d_head, d_latent1)
        absorbed_q = einsum(q, W_UK, "b h s d, h d l -> b h s l")                   # shape: (B, H, (T/1), d_latent1)

        # -------------------------------- XX -------------------------------- 
        # If RoPE is used: (kR, qR, reshape tensors, apply RoPE)
        # -------------------------------- XX -------------------------------- 
        if self.rope is not None:

            # -------------------------------- XX -------------------------------- 
            # Get kR and qR
            # -------------------------------- XX -------------------------------- 
            kR = self.proj_kR(x)        # shape: (B, (T/1), d_headR)
            qR = self.proj_qR(latent_q) # shape: (B, (T/1), n_heads*d_headR)

            # -------------------------------- XX -------------------------------- 
            # Reshape tensors across heads
            # -------------------------------- XX -------------------------------- 
            kR = rearrange(kR, "b s (h dR) -> b h s dR", dR=self.d_headR)    # shape: (B, 1, (T/1), d_headR)
            qR = rearrange(qR, "b s (h dR) -> b h s dR", dR=self.d_headR)    # shape: (B, H, (T/1), d_headR)

            # -------------------------------- XX -------------------------------- 
            # Apply RoPE
            # -------------------------------- XX -------------------------------- 
            if use_mhla_cache:
                kR = self.rope.apply_rope(x=kR, seq_offset=mhla_cache.ntokens_processed)
                qR = self.rope.apply_rope(x=qR, seq_offset=mhla_cache.ntokens_processed)
            else:
                kR = self.rope.apply_rope(x=kR)
                qR = self.rope.apply_rope(x=qR)  

  
        # -------------------------------- XX -------------------------------- 
        # MHLA Caching: Update the latent_kv and kR vectors (current token)
        # -------------------------------- XX --------------------------------
        if use_mhla_cache:

            # Resolve Prefill/Decode stage (this controls behaviour for causal masking)
            is_decode = mhla_cache.latent_kv_cache is not None

            # Update the Latent KV Cache 
            mhla_cache.update_cache(latent_kv, is_latent=True)

            if self.rope is not None:

                # Update the kR
                mhla_cache.update_cache(kR, is_latent=False)


        # -------------------------------- XX -------------------------------- 
        # MHLA Caching: Retrieve the complete latent_kv and kR vectors (current + past tokens)
        # Retrieval should be done after complete update of both latent_kv and kR
        # Otherwise states-curr_idx can cause problem
        # -------------------------------- XX --------------------------------
        if use_mhla_cache:

            # -------------------------------- XX --------------------------------
            # Retrieve: latent_kv_cache
            # -------------------------------- XX --------------------------------
            # Prefill Stage: Update the latent_kv vector corresponding to all q vectors from prompt tokens, then fetch entire latent_kv (overrwriting)
            # Decode Stage : Update the latent_kv vector corresponding to current q (current token), then fetch entire latent_kv (past + current token)
            latent_kv = mhla_cache.latent_kv_cache[:, :mhla_cache.curr_idx]       # shape: (B, T, d_latent1)

            if self.rope is not None:

                # -------------------------------- XX --------------------------------
                # Retrieve: roped_key_cache and repeat kR to be shared across n heads
                # -------------------------------- XX --------------------------------
                # Prefill Stage: Update the kR vector corresponding to all q vectors from prompt tokens, then fetch entire kR (overrwriting)
                # Decode Stage : Update the kR vector corresponding to current q (current token), then fetch entire kR (past + current token)
                kR = mhla_cache.roped_key_cache[:, :, :mhla_cache.curr_idx]       # shape: (B, 1, T, d_headR)
                kR = repeat(kR, "b g s dR -> b (g n) s dR", n=self.n_heads)       # shape: (B, H, T, d_headR)

        # -------------------------------- XX -------------------------------- 
        # Get V and reshape V across heads (After latent_kv retrieved from cache)
        # -------------------------------- XX -------------------------------- 
        v = self.proj_up_v(latent_kv)                                # shape: (B, T, H*d_head)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)     # shape: (B, H, T, d_head)


        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Flag to control causal masking behaviour
        # -------------------------------- XX -------------------------------- XX --------------------------------
        is_causal_flag=True             # Usual Flow (Without Cache/With Cache:Train & Infer-Prefill)
        if use_mhla_cache and is_decode:
            is_causal_flag=False        # Overrides flag to false only when Cache is used + Infer-Decode

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Attention
        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Attention_p1.1: Latent Attention
        # Shape: (B, H, T, T) = (B, H, T, d_latent1) @ (B, T, d_latent1).T
        attn_latent = einsum(absorbed_q, latent_kv, "b h si l, b sj l -> b h si sj") 

        # Attention_p1.2: Rope Attention
        if self.rope is not None:
            # Shape: (B, H, T, T) = (B, H, T, d_headR) @ (B, H, T, d_headR).T
            attn_R = einsum(qR, kR, "b h si dR, b h sj dR -> b h si sj") 

        # Attention_p1: Attention scores
        if self.rope is not None:
            attn_scores = attn_latent + attn_R
        else:
            attn_scores = attn_latent

        # Attention_p2: Scaling  
        attn_scores *= (1/((self.d_head+self.d_headR)**0.5))

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


class GroupedQueryAttention(nn.Module):
    
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.d_model=config.d_model
        self.n_heads=config.n_heads
        self.n_groups=config.n_groups
        self.head_dim=(config.d_model//config.n_heads)
        # QKV projection layer
        self.proj_qkv = nn.Linear(config.d_model, config.d_model + 2*self.n_groups*self.head_dim)
        # Output projection layer
        self.proj_out = nn.Linear(config.d_model, config.d_model)
        self.proj_out.RESIDUAL_PATH_SCALE_INIT=1
        # Rotary Embedding
        self.rope = RoPE(head_dim=self.head_dim) if config.rotary_embedding else None
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
        q, k, v = torch.split(qkv, [self.d_model, self.n_groups*self.head_dim, self.n_groups*self.head_dim], dim=-1)

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
                # This path used during Inference (Prefill + Decode) with KV Cache enabled
                # Handles sequences during inference time which are longer than ctx_len supported
                # Slides context of the last ctx_len tokens
                # Bake absolute postion into KV cache by rotating by m (pos of current token in seq)
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
        # Repeat k v to be shared across n heads 
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
        is_causal_flag=True             # Usual Flow (Without Cache/With Cache:Train & Infer-Prefill)
        if use_kv_cache and is_decode:
            is_causal_flag=False        # Overrides flag to false only when Cache is used + Infer-Decode

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Attention
        # -------------------------------- XX -------------------------------- XX --------------------------------
        y, self.attn_debug_probs = compute_attention(
            q=q, 
            k=k, 
            v=v, 
            use_flash=self.use_flash,
            dropout_p=self.dropout_p if self.training else 0, 
            is_causal_flag=is_causal_flag, 
            head_dim=(self.d_model//self.n_heads),
            attn_debug=self.attn_debug,
            dropout_attn=self.dropout_attn,
            device=x.device
        )

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
        self.d_model=config.d_model
        self.n_heads=config.n_heads
        self.head_dim=(config.d_model//config.n_heads)
        # QKV projection layer
        self.proj_qkv=nn.Linear(config.d_model, 3*config.d_model)
        # Output projection layer
        self.proj_out=nn.Linear(config.d_model, config.d_model)
        self.proj_out.RESIDUAL_PATH_SCALE_INIT=1
        # Rotary Embedding
        self.rope = RoPE(head_dim=self.head_dim) if config.rotary_embedding else None
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
        is_causal_flag=True             # Usual Flow (Without Cache/With Cache:Train & Infer-Prefill)
        if use_kv_cache and is_decode:
            is_causal_flag=False        # Overrides flag to false only when Cache is used + Infer-Decode

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Attention
        # -------------------------------- XX -------------------------------- XX --------------------------------
        y, self.attn_debug_probs = compute_attention(
            q=q, 
            k=k, 
            v=v, 
            use_flash=self.use_flash,
            dropout_p=self.dropout_p if self.training else 0, 
            is_causal_flag=is_causal_flag, 
            head_dim=self.head_dim,
            attn_debug=self.attn_debug,
            dropout_attn=self.dropout_attn,
            device=x.device
        )

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
            return MultiHeadLatentAttention(config)


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

    # MHA
    d_model=32
    n_heads=8
    use_flash=False
    attn_debug=True
    attention='mhla'
    
    # GQA (MHA settings + GQA config)
    n_groups=2

    # MHLA (MHA settings + GQA config)
    d_latent1=10
    d_latent2=6
    d_headR=2

    config = LLMConfig(
        d_model=d_model, 
        attention=attention,
        n_heads=n_heads, 
        n_groups=n_groups,
        use_flash=use_flash, 
        attn_debug=attn_debug,
        rotary_embedding=False,
        d_latent1=d_latent1,
        d_latent2=d_latent2,
        d_headR=d_headR
    )
    mha  = MultiHeadAttention(config)
    gqa  = GroupedQueryAttention(config)
    mhla = MultiHeadLatentAttention(config)


    x = torch.randn(2, 20, d_model)
    print('x.shape:\n', x.shape)
    x_mha = mha(x)
    x_gqa = gqa(x)
    x_mhla = mhla(x)
    print('x_mha.shape:\n', x_mha.shape)
    print('x_gqa.shape:\n', x_gqa.shape)
    print('x_mhla.shape:\n', x_mhla.shape)

    # if mha.attn_debug_probs is not None:
    #     print(mha.attn_debug_probs.shape)
    #     visualize_attnscores(mha.attn_debug_probs, mha.n_heads)

    # if gqa.attn_debug_probs is not None:
    #     print(gqa.attn_debug_probs.shape)
    #     visualize_attnscores(gqa.attn_debug_probs, gqa.n_heads)

    # if mhla.attn_debug_probs is not None:
    #     print(mhla.attn_debug_probs.shape)
    #     visualize_attnscores(mhla.attn_debug_probs, mhla.n_heads)

