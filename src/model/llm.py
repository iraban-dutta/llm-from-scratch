import torch
import torch.nn as nn
import torch.nn.functional as F
from .llm_config import LLMConfig
from .position_embedding import build_position_embedding
from .layers import Decoder
from .normalization import LayerNorm

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
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)


    def forward(self, x:torch.Tensor, y:None|torch.Tensor=None) -> torch.Tensor:

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

        # Forward Pass
        x = self.transformer.wte(x)
        x = self.transformer.wpe(x)
        x = self.transformer.drp(x)
        for decoder in self.transformer.dec:
            x=decoder(x)
        x = self.transformer.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))

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
        attn_debug=False
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
    print(f"Model instantiated and moved to {device}")
    print('-'*50)
    
    # Total params
    total_params = sum(p.nelement() for p in model.parameters())
    print(f'Total parameters:\n {(total_params/1e6):.2f}M')
    print('-'*50)

    # Param contribution
    print(f"{'Param':45s} {'Shape':15s} {'Contribution':10s}")
    for k,v in model.state_dict().items():
        print(f"{k:45s} {str(tuple(v.shape)):15s} {(100.0*v.nelement()/total_params):10.2f}%")
    print('-'*50)

    # Access different layers of model
    # print(model.transformer)
    # print(model.transformer.wte)
    # print(model.transformer.dec)

    # Dummy Forward pass of a mini batch
    g=torch.Generator(device=device).manual_seed(42)
    x = torch.randint(low=0, high=50256, size=(16, ctx_len), generator=g, device=device)
    print(x.shape, x.device)
    x, _ = model(x)
    print(x.shape, x.device)
    print('-'*50)


