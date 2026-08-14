from dataclasses import dataclass
import torch
import torch.nn.functional as F
from src.model.llm_config import LLMConfig
from src.model.llm import LLM
from config.test import SAMPLING_STRATEGIES, TOKENIZERS_SUPPORTED


class Sampler:
    def __init__(self, strategy:str, temperature:float, top_k:int):

        # Validate temperature
        if temperature <= 0:
            raise ValueError(
                f"Temperature must be > 0, got {temperature}"
            )
        # Validate strategy
        assert strategy in SAMPLING_STRATEGIES, (
            f"Invalid sampling strategy: {strategy}. Sampling strategies supported: {SAMPLING_STRATEGIES}"
        )

        self.temperature=temperature
        self.top_k=top_k
        self.strategy=strategy
        self._sampling_function = {
            'greedy': self._greedy,
            'random': self._random,
            'topk'  : self._topk 
        }[strategy]

    def sample_next_token(
        self, 
        logits:torch.Tensor, 
        generator:torch.Generator|None = None) -> torch.Tensor:
        # logits.shape = (B, T, vocab_size)
        # out.shape    = (B, 1)
        return self._sampling_function(logits, generator)

    def _greedy(
        self, 
        logits:torch.Tensor, 
        generator:torch.Generator|None = None) -> torch.Tensor:

        # Get the last logits: Shape = (B, vocab_size)
        logits_last = logits[:, -1, :]

        # Argmax over vocab_size
        next_token = logits_last.argmax(dim=-1, keepdim=True)

        return next_token

    def _random(
        self, 
        logits:torch.Tensor, 
        generator:torch.Generator|None = None) -> torch.Tensor:

        # Get the last logits: Shape = (B, vocab_size)
        logits_last = logits[:, -1, :]

        # Adjust for temperature
        logits_last = (1/self.temperature) * logits_last

        # Softmax: Shape = (B, vocab_size)
        probs_last = F.softmax(logits_last, dim=-1)

        # Sample: Shape = (B, 1)
        next_token = torch.multinomial(probs_last, num_samples=1, generator=generator)

        return next_token

    def _topk(
        self, 
        logits:torch.Tensor,
        generator:torch.Generator|None = None) -> torch.Tensor:
                
        # Get the last logits: Shape = (B, vocab_size)
        logits_last = logits[:, -1, :]

        # Adjust for temperature
        logits_last = (1/self.temperature) * logits_last

        # Get topk idxs
        topk_vals, topk_idxs = torch.topk(logits_last, k=self.top_k, dim=-1)

        # Filtered logits with only topk values: Shape = (B, vocab_size)
        logits_filtered_topk = torch.full_like(logits_last, fill_value=-torch.inf)
        logits_filtered_topk.scatter_(dim=1, index=topk_idxs, src=topk_vals) # In-place update

        # Softmax: Shape = (B, vocab_size)
        probs_last = F.softmax(logits_filtered_topk, dim=-1)

        # Sample: Shape = (B, 1)
        next_token = torch.multinomial(probs_last, num_samples=1, generator=generator)

        return next_token


@dataclass
class TextGeneratorConfig:
    model:LLM
    num_samples:int
    max_new_tokens:int
    tokenizer:str='gpt2'
    strategy:str='topk'
    temperature:float=1.0
    top_k:int=50
    

class TextGenerator:
    def __init__(self, config:TextGeneratorConfig):
        self.model=config.model
        self.model.eval()
        self.num_samples=config.num_samples
        self.max_new_tokens=config.max_new_tokens

        # Validate tokenizer
        assert config.tokenizer in TOKENIZERS_SUPPORTED, (
            f"Invalid tokenizer: {config.tokenizer}. Tokenizers supported: {TOKENIZERS_SUPPORTED}"
        ) 
        self.enc = None
        # Define Tokenizer
        if config.tokenizer=='gpt2':
            import tiktoken
            self.enc=tiktoken.get_encoding(config.tokenizer) 

        # Define sampler
        self.sampler = Sampler(config.strategy, config.temperature, config.top_k)

    def _encode_prompt(self, prompt:str) -> torch.Tensor:
        return torch.tensor(self.enc.encode(prompt)).unsqueeze(dim=0)

    def _decode_tokens(self, x:torch.Tensor) -> str:
        return self.enc.decode(x)
        
    def generate(self, prompt:str, generator:None|torch.Generator = None) -> list[str]:

        # Encode prompt into tokens: x.shape = (1, T)
        x = self._encode_prompt(prompt)

        # Expand x from (1, T) to (num_samples, T)
        x_tokens = x.expand(self.num_samples, -1)

        # Resolve device and move x_tokens to device
        x_tokens = x_tokens.to(next(self.model.parameters()).device)

        with torch.no_grad():
            # AutoRegressive Generation
            for i in range(self.max_new_tokens):

                # If max ctx_len is reached, then slide context over the latest tokens
                x_tokens_fwd = x_tokens[:, -(self.model.config.ctx_len):]

                # Model Forward pass: Shape = (num_samples, T, vocab_size)
                logits, _ = self.model(x_tokens_fwd)

                # Next token: Shape = (num_samples, 1)
                next_token = self.sampler.sample_next_token(logits, generator)

                # Append to old prompt: Shape = (num_samples, T+i)
                x_tokens = torch.cat([x_tokens, next_token], dim=-1)

        out = list(map(lambda x: self._decode_tokens(x.tolist()), x_tokens))
        return out    


if __name__=='__main__':

    # ======== DEFINE Model ========
    ctx_len = 32
    d_model = 64
    
    llm_config = LLMConfig(
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
    print(llm_config)
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
    model = LLM(llm_config)
    model = model.to(device)
    print(f"Model instantiated and moved to {device}")
    print('-'*50)


    # ======== DEFINE Generator ========
    text_gen_config = TextGeneratorConfig(
        model=model,
        num_samples=5,
        max_new_tokens=10,
        tokenizer='gpt2',
        strategy='topk',
        temperature=1.0,
        top_k=50
    )

    # Instantiate generator
    text_generator = TextGenerator(text_gen_config)


    # Generate
    prompt = "Hey, hi"
    g = torch.Generator(device=device).manual_seed(42)
    out = text_generator.generate(prompt, generator=g)

    for sample in out:
        print(sample)
        print('-'*50)






