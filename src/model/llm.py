import torch
import torch.nn as nn
import torch.nn.functional as F
from .llm_config import LLMConfig
from .position_embedding import build_position_embedding
from .layers import Decoder
from .normalization import LayerNorm
from src.inference.cache import KVCacheManager, MHLACacheManager

class LLM(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.config=config

        model_dict = {
            "wte":nn.Embedding(config.vocab_size, config.d_model),
            "wpe":build_position_embedding(config),
            "drp":nn.Dropout(config.dropout),
            "dec": nn.ModuleList([Decoder(config) for _ in range(config.n_layer)]),
            "ln_final":LayerNorm(config)

        }
        
        self.transformer = nn.ModuleDict(model_dict)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, config.bias)

        # Weight Tying
        self.transformer.wte.weight = self.lm_head.weight
        assert self.transformer.wte.weight.data_ptr() == self.lm_head.weight.data_ptr(), (
            f"Issue in Weight Tying"
        )
        print('Weight Tying Done: B/W initial Embedding Layer and final Linear Layer before Softmax!')

        # Manual init
        self.apply(self._init_weights)

        # Report #params for model
        print(f"Model initialized, number of parameters: {(self._get_num_params()/1e6):.2f}M")

    def _init_weights(self, module):

        std = 0.02
        if isinstance(module, nn.Linear) and hasattr(module, 'RESIDUAL_PATH_SCALE_INIT'):
            std = 1/((2*self.config.n_layer)**0.5)
            torch.nn.init.normal_(tensor=module.weight, mean=0.0, std=std)
        
        # For nn.Linear.weight: keep init scaling - default
        # For nn.Linear.bias  : keep init scaling - as zero
        if isinstance(module, nn.Linear):
            # torch.nn.init.normal_(tensor=module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(tensor=module.bias)

        # For nn.Embedding.weight: keep init scaling - default
        # if isinstance(module, nn.Embedding):
        #     torch.nn.init.normal_(tensor=module.weight, mean=0.0, std=std)



    def _get_num_params(self, non_embedding:bool=True) -> float:

        # Get total count of model params - include/exclude positional embedding params (only for LearnedPostionalEmbedding)
        total_params = sum(p.nelement() for p in self.parameters())
        pos_emb_params = sum(p.nelement() for p in self.transformer.wpe.parameters())
        if pos_emb_params>0 and non_embedding:
            total_params -= pos_emb_params
        return total_params


    def forward(self, x:torch.Tensor, y:None|torch.Tensor=None, cache_manager:KVCacheManager|MHLACacheManager|None=None) -> torch.Tensor:

        # x.shape = (B, T)
        B, T = x.shape

        # Resolve Device and move input to device
        x = x.to(next(self.parameters()).device)
        # print(f"Moved input to {device}")

        # Check for context overflow: T <= ctx_len
        assert T<=self.config.ctx_len, (
                f"Input shape: {tuple(x.shape)}"
                f"Sequence length in x={T} should be lesser than max context length of model {self.config.ctx_len}"
        )

        use_cache = cache_manager is not None

        # Forward Pass1: Token Embeddings
        x = self.transformer.wte(x)

        # Forward Pass2: Absolute Positional Embeddings (If RoPE: Bypassed with nn.Identity)
        if use_cache:
            # Get the current token idx (During Inference: Prefill or Decode)
            curr_token_idx = cache_manager[0].curr_idx

            # In Prefill: We ensure x is of shape (B, T, d_model) and T is always upper bounded to ctx_len
            # In Decode : We get x of shape (B, 1, d_model) -> Once ctx_len is full, we slide the context over the latest tokens in window ctx_len
            # In Decode : Once the ctx_len is full, we always eject the oldest token from KV Cache and append the current token to the last position of cache
            curr_token_idx = min(curr_token_idx, self.config.ctx_len-1)

            # Get PEs (Adjusts for the position of current token in the sequence being decoded)
            x = self.transformer.wpe(x, position_offset=curr_token_idx)
        else:
            # Normal Path: Without KV Cache
            x = self.transformer.wpe(x, position_offset=0)

        # Forward Pass3: Dropout
        x = self.transformer.drp(x)

        # Forward Pass4: Sequence of Decoder Layers (Norm + Attention + Norm + MLP)
        for i, decoder in enumerate(self.transformer.dec):
            cache = None
            if use_cache:
                cache = cache_manager[i]
            x=decoder(x, cache)


        # Forward Pass5: Norm (Before LM Head)
        x = self.transformer.ln_final(x)

        # Forward Pass6: LM head
        logits = self.lm_head(x)

        loss = None
        if y is not None:
            # Language Modelling CCE Loss
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            # MoE auxiliary losses
            if self.config.use_moe:
                aux_loss = 0.0
                for decoder in self.transformer.dec:
                    aux_loss += decoder.mlp.loss_coeff_var
                    aux_loss += decoder.mlp.loss_load_balance

                loss += aux_loss

        return logits, loss


if __name__=='__main__':

    ctx_len = 32
    d_model = 64
    
    config = LLMConfig(
        vocab_size=50257,
        ctx_len=ctx_len,
        d_model=d_model, 
        n_layer=2,
        ff_ratio=4,
        dropout=0.0,
        eps=1e-5,
        bias=False,
        position_embedding='sinusoidal',
        rotary_embedding=False,
        attention='mha',
        normalization='layernorm',
        n_heads=4, 
        n_groups=None,
        use_flash=False, 
        attn_debug=False,
        use_moe=True,
        n_experts=3,
        n_shared_experts=1,
        topk=2,
        capcity_factor = 1.2,
        noisy_router=False,
        router_noise_std=0.0,
        scale_aux_loss_expert_imp=1.0,
        scale_aux_loss_load_balance=1.0,
        aux_loss_free_load_balance=False,
        aux_loss_free_load_balance_bias_update=0.0
    )
    print(config)
    print('-'*50)

    # Detect and resolve device
    device = 'cpu'
    if torch.cuda.is_available():
        device='cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device='mps'

    print("Device found:", device)
    print('-'*50)


    # Instantiate model
    model = LLM(config)
    model = model.to(device)
    print(f"Model moved to {device}")
    print('-'*50)
    
    # Total params
    total_params = model._get_num_params()

    # Param contribution
    print(f"{'Param':50s} {'Shape':15s} {'Contribution':10s}")
    for k,v in model.state_dict().items():
        print(f"{k:50s} {str(tuple(v.shape)):15s} {(100.0*v.nelement()/total_params):10.2f}%")
    print('-'*50)

    # Access different layers of model
    # print(model.transformer)
    # print(model.transformer.wte)
    # print(model.transformer.dec)

    # # Debug Breakpoint
    # breakpoint()

    # Dummy Forward pass of a mini batch
    g=torch.Generator(device=device).manual_seed(42)
    x = torch.randint(low=0, high=50256, size=(16, ctx_len), generator=g, device=device)
    y = torch.randint(low=0, high=50256, size=(16, ctx_len), generator=g, device=device)
    print(x.shape, x.device)
    print(y.shape, y.device)
    x, _ = model(x, y)
    print(x.shape, x.device)
    print(y.shape, y.device)
    print('-'*50)


