import argparse
import importlib
import time
from pathlib import Path

import numpy as np
import torch

from config.constants import SAMPLING_STRATEGIES, TOKENIZERS_SUPPORTED
from src.inference.generate import TextGenerator, TextGeneratorConfig
from src.model.llm import LLM


# ==============================================================================
# Device
# ==============================================================================
def resolve_device(device: str) -> torch.device:
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def synchronize(device: torch.device) -> None:
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


# ==============================================================================
# Generation helpers
# ==============================================================================
def print_samples(samples, title):
    print('=' * 32)
    print(title)
    print('=' * 32)
    for sample in samples:
        print(sample)
        print('-' * 50)


def run_generation(text_generator, prompt, generator, naive=False):
    if naive:
        return text_generator.generate_naive(prompt, generator=generator)
    return text_generator.generate(prompt, generator=generator)


def print_benchmark_summary(label, elapsed_s, n_tokens):
    print(
        f'Finished Benchmark: {label:<10}  '
        f'Total Time: {elapsed_s:6.2f} s for {n_tokens} tokens, '
        f'Mean(Tokens/Sec): {n_tokens / elapsed_s:7.2f}'
    )


# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(
    description='Generate text from a trained checkpoint or a config (random weights).',
)
parser.add_argument(
    '--ckpt',
    default=None,
    help='Path to checkpoint .pt — model config and weights come from the checkpoint',
)
parser.add_argument(
    '--config',
    default='config.dense_default',
    help='Config module when no checkpoint is provided, e.g. config.dense_default',
)
parser.add_argument(
    '--prompt',
    default='Once upon a time',
    help='Prompt text to continue',
)
parser.add_argument('--num-samples', type=int, default=5)
parser.add_argument('--max-new-tokens', type=int, default=50)
parser.add_argument(
    '--tokenizer',
    default='gpt2',
    choices=TOKENIZERS_SUPPORTED,
)
parser.add_argument(
    '--strategy',
    default='topk',
    choices=SAMPLING_STRATEGIES,
)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top-k', type=int, default=50)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument(
    '--device',
    default='auto',
    help='auto | cpu | cuda | mps',
)
parser.add_argument(
    '--naive',
    action='store_true',
    help='Use generate_naive (no KV cache)',
)
parser.add_argument(
    '--benchmark',
    action='store_true',
    help='Time cached vs naive generation',
)
args = parser.parse_args()


# ==============================================================================
# Resolve device + seed
# ==============================================================================
device = resolve_device(args.device)
torch.manual_seed(args.seed)
print(f'Device: {device}')
print('-' * 50)


# ==============================================================================
# Mode 1: Load model from checkpoint
# ==============================================================================
if args.ckpt:
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    print(f'Inference from checkpoint: {ckpt_path}')
    print('-' * 50)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    llm_config = ckpt['model_config']
    print(llm_config)
    print('-' * 50)

    model = LLM(config=llm_config)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded weights from {ckpt_path}')
    print('-' * 50)

# ==============================================================================
# Mode 2: Fresh model from config file (random weights)
# ==============================================================================
else:
    print(f'Inference with config (random weights): {args.config}')
    print('-' * 50)

    cfg = importlib.import_module(args.config)
    llm_config = cfg.llm_config
    print(llm_config)
    print('-' * 50)

    model = LLM(config=llm_config)
    print('No --ckpt provided; generating with randomly initialized weights')
    print('-' * 50)


# ==============================================================================
# Move model to device and set to evaluation mode
# ==============================================================================
model = model.to(device)
model.eval()
print(f'Model moved to {device}')
print('-' * 50)


# ==============================================================================
# Define generator
# ==============================================================================
text_gen_config = TextGeneratorConfig(
    model=model,
    num_samples=args.num_samples,
    max_new_tokens=args.max_new_tokens,
    tokenizer=args.tokenizer,
    strategy=args.strategy,
    temperature=args.temperature,
    top_k=args.top_k,
)
print(
    f'num_samples={args.num_samples}, max_new_tokens={args.max_new_tokens}, '
    f'tokenizer={args.tokenizer}, strategy={args.strategy}, '
    f'temperature={args.temperature}, top_k={args.top_k}, seed={args.seed}'
)
print('-' * 50)

text_generator = TextGenerator(text_gen_config)
g = torch.Generator(device=device).manual_seed(args.seed)


# ==============================================================================
# Generate
# ==============================================================================
print('Starting Generation...')
print('-' * 50)
print(f'Prompt: {args.prompt}')
print('-' * 50)

if args.benchmark:
    start_time = time.perf_counter()
    out_cached = run_generation(text_generator, args.prompt, g, naive=False)
    synchronize(device)
    gen_t = time.perf_counter() - start_time
    print_samples(out_cached, 'GENERATED SAMPLES')

    g_naive = torch.Generator(device=device).manual_seed(args.seed)
    start_time_naive = time.perf_counter()
    out_naive = run_generation(text_generator, args.prompt, g_naive, naive=True)
    synchronize(device)
    gen_t_naive = time.perf_counter() - start_time_naive
    print_samples(out_naive, 'GENERATED SAMPLES: NAIVE')

    n_tokens = args.num_samples * args.max_new_tokens
    print_benchmark_summary('With Cache', gen_t, n_tokens)
    print_benchmark_summary('Naive', gen_t_naive, n_tokens)
    print('-' * 50)
else:
    out = run_generation(text_generator, args.prompt, g, naive=args.naive)
    title = 'GENERATED SAMPLES: NAIVE' if args.naive else 'GENERATED SAMPLES'
    print_samples(out, title)
