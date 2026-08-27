import time
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .cache import KVCacheManager, MHLACacheManager
from src.model.llm_config import LLMConfig
from src.model.llm import LLM
from config.test import SAMPLING_STRATEGIES, TOKENIZERS_SUPPORTED


def get_n_kv_heads(config):
    if config.attention == "mha":
        return config.n_heads
    elif config.attention == "gqa":
        return config.n_groups
    elif config.attention == "mhla":
        raise NotImplementedError("MHLA KV cache not implemented yet")
    else:
        raise ValueError(f"Unknown attention type: {config.attention}")


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

        # Cache Manager
        if self.model.config.attention=='mhla':
            # MHLA Cache Manager: Works with MHLA
            self.cache_manager = MHLACacheManager(
                d_latent=self.model.config.d_latent1,
                d_headR=self.model.config.d_headR,
                max_new_tokens=self.max_new_tokens,
                ctx_len=self.model.config.ctx_len,
                is_rope=self.model.config.rotary_embedding,
                n_layer=self.model.config.n_layer
            )
        elif self.model.config.attention in ('mha', 'gqa'):
            # KV Cache Manager: Works with MHA and GQA
            self.cache_manager = KVCacheManager(
                head_dim=(self.model.config.d_model//self.model.config.n_heads),
                n_kv=get_n_kv_heads(self.model.config),
                max_new_tokens=self.max_new_tokens,
                ctx_len=self.model.config.ctx_len,
                n_layer=self.model.config.n_layer
            )


    def _encode_prompt(self, prompt:str) -> torch.Tensor:
        return torch.tensor(self.enc.encode(prompt)).unsqueeze(dim=0)

    def _decode_tokens(self, x:torch.Tensor) -> str:
        return self.enc.decode(x)


    def generate_naive(self, prompt:str, generator:None|torch.Generator = None) -> list[str]:

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


        
    def generate(self, prompt:str, generator:None|torch.Generator = None) -> list[str]:

        # Encode prompt into tokens: x.shape = (1, T)
        x = self._encode_prompt(prompt)

        # Expand x from (1, T) to (num_samples, T)
        x_tokens = x.expand(self.num_samples, -1)

        # Resolve device and move x_tokens to device
        x_tokens = x_tokens.to(next(self.model.parameters()).device)

        with torch.no_grad():

            # ======== Prefill ========
            # Model Forward pass: Shape = (num_samples, T, vocab_size)
            logits, _ = self.model(x=x_tokens, cache_manager=self.cache_manager)
            # print(self.cache_manager[0].ntokens_processed)

            # Next token: Shape = (num_samples, 1)
            next_token = self.sampler.sample_next_token(logits, generator)

            # Append to x_tokens
            x_tokens=torch.cat([x_tokens, next_token], dim=-1)

            # ======== Decode: AutoRegressive Generation ========
            for i in range(self.max_new_tokens-1):

                # Model Forward pass: Shape = (num_samples, 1, vocab_size)
                logits, _ = self.model(x=next_token, cache_manager=self.cache_manager)
                # print(self.cache_manager[0].ntokens_processed)

                # Next token: Shape = (num_samples, 1)
                next_token = self.sampler.sample_next_token(logits, generator)

                # Append to old prompt: Shape = (num_samples, T+i)
                x_tokens = torch.cat([x_tokens, next_token], dim=-1)

        # Reset Cache
        self.cache_manager.reset()

        # De-Tokenize
        out = list(map(lambda x: self._decode_tokens(x.tolist()), x_tokens))
        return out    


if __name__=='__main__':

    # ======== DEFINE Model ========    
    T = 128
    
    llm_config = LLMConfig(
        vocab_size=50304,     # Divisible by 128
        ctx_len=T,
        d_model=512, 
        n_layer=8,
        ff_ratio=4,
        dropout=0.0,
        eps=1e-5,
        bias=False,
        position_embedding='sinusoidal',
        rotary_embedding=True,
        attention='mhla',
        normalization='layernorm',
        n_heads=8, 
        # n_groups=4,
        d_latent1=(512//8)*2,
        d_latent2=(512//8)*2,
        d_headR=(512//8)//2,
        use_flash=False, 
        attn_debug=False
    )
    print(llm_config)
    print('-'*50)


    # Detect and resolve device
    device=torch.device('cpu')
    if torch.cuda.is_available():
        device=torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device=torch.device('mps')

    print("Device found:", device)
    print('-'*50)


    # Instantiate model
    model = LLM(llm_config)
    model = model.to(device)
    print(f"Model instantiated and moved to {device}")
    print('-'*50)

    # # CKPT
    # ckpt_path = '/Users/irabandutta/Developer/2026-08-llm-from-scratch/checkpoints/2026_08_23_03_48_39/best.pt'
    # ckpt = torch.load(ckpt_path, weights_only=False)
    # print('Checkpoint Loaded')
    # model.load_state_dict(ckpt['model_state_dict'])


    # ======== DEFINE Generator ========
    SAMPLES = 5
    MAX_NEW_TOKENS = 50

    text_gen_config = TextGeneratorConfig(
        model=model,
        num_samples=SAMPLES,
        max_new_tokens=MAX_NEW_TOKENS,
        tokenizer='gpt2',
        strategy='topk',
        temperature=1.0,
        top_k=50
    )

    # Instantiate generator
    text_generator = TextGenerator(text_gen_config)


    # # ================================
    # # DEBUG: START 
    # # ================================
    # prompt = "Hey, hi"
    # x = text_generator._encode_prompt(prompt).to(device)

    # # Naive
    # logits_naive, _ = model(x)

    # # KV prefill
    # text_generator.cache_manager.reset()
    # logits_prefill, _ = model(
    #     x,
    #     cache_manager=text_generator.cache_manager
    # )

    # print(
    #     "PREFILL:",
    #     (logits_naive[:, -1] - logits_prefill[:, -1]).abs().max()
    # )


    # next_token = logits_naive[:, -1:].argmax(-1)

    # # Naive: prompt + token
    # x2 = torch.cat([x, next_token], dim=1)
    # logits_naive_2, _ = model(x2)

    # # Cached decode
    # logits_cached_2, _ = model(
    #     next_token,
    #     cache_manager=text_generator.cache_manager
    # )

    # print(
    #     "DECODE:",
    #     (logits_naive_2[:, -1] - logits_cached_2[:, -1]).abs().max()
    # )
    # # ================================
    # # DEBUG: END 
    # # ================================


    # ======== Generation with Cache ========
    # Prompt
    prompt = "Hey, hi"
    g = torch.Generator(device=device).manual_seed(42)
    tokens_per_sec_hist = []
    start_time = time.perf_counter()

    # Generate: Call method to run Prefill + Decode
    out = text_generator.generate(prompt, generator=g)

    # Synchronize
    if device.type == 'mps':
        torch.mps.synchronize()

    end_time = time.perf_counter()

    print("================================")
    print("GENERATED SAMPLES")
    print("================================")
    for sample in out:
        print(sample)
        print('-'*50)


    # ======== NAIVE Generation ========
    # Prompt
    prompt = "Hey, hi"
    g = torch.Generator(device=device).manual_seed(42)
    tokens_per_sec_hist_naive = []
    start_time_naive = time.perf_counter()

    # Generate: Call method to run Prefill + Decode
    out = text_generator.generate_naive(prompt, generator=g)

    # Synchronize
    if device.type == 'mps':
        torch.mps.synchronize()

    end_time_naive = time.perf_counter()

    print("================================")
    print("GENERATED SAMPLES: NAIVE")
    print("================================")
    for sample in out:
        print(sample)
        print('-'*50)

    # Generation Time
    gen_t = end_time - start_time
    gen_t_naive = end_time_naive - start_time_naive

    # Throughput
    tokens_per_sec_hist.append((SAMPLES*MAX_NEW_TOKENS)/gen_t)
    tokens_per_sec_hist_naive.append((SAMPLES*MAX_NEW_TOKENS)/gen_t_naive)
    print(f'Finished Benchmark:With Cache, Total Time: {gen_t:.2f} ms for {SAMPLES*MAX_NEW_TOKENS} tokens, Mean(Tokens/Sec): {np.mean(tokens_per_sec_hist):.2f}')
    print(f'Finished Benchmark:Naive, Total Time: {gen_t_naive:.2f} ms for {SAMPLES*MAX_NEW_TOKENS} tokens, Mean(Tokens/Sec): {np.mean(tokens_per_sec_hist_naive):.2f}')
    print('-'*50)








