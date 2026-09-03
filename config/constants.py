import numpy as np

# Training / data paths
TRAIN_BIN = "./data/tinystories/processed/train.bin"
VAL_BIN = "./data/tinystories/processed/val.bin"
TOKEN_DTYPE = np.uint16
VOCAB_SIZE = 50304     # Divisible by 128

# Inference / preprocessing
SAMPLING_STRATEGIES = ['greedy', 'random', 'topk']
TOKENIZERS_SUPPORTED = ['gpt2']
