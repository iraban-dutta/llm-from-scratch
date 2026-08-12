from pathlib import Path
import json
import numpy as np
from .tokenizer import tokenize_tinystories
from config.test import TOKENIZERS_SUPPORTED


def iter_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def tokenize_save_binary(loadfile_path:Path, savefile_name:str, chunk_size:int=100, tokenizer:str='gpt2') -> None:

    assert tokenizer in TOKENIZERS_SUPPORTED, (
        f"Invalid tokenizer: {tokenizer}. Tokenizers supported: {TOKENIZERS_SUPPORTED}"
    )

    print('='*50)
    print(f'Tokenizing {loadfile_path.name} and saving in .bin format')
    print('='*50)

    # Create savefile_dir
    savefile_dir = loadfile_path.parent.parent/f'processed/'
    savefile_dir.mkdir(parents=True, exist_ok=True)

    # Create savefile_path
    savefile_path = savefile_dir/f'{savefile_name}'
    
    if savefile_path.exists():
        print("Binary files exist - tokenizing and saving skipped")
    else:

        # Define Tokenizer
        if tokenizer=='gpt2':
            import tiktoken
            enc=tiktoken.get_encoding(tokenizer)

        # Tokenize and save binary file in chunks
        
        print(f'Tokenizing and saving .bin file chunk by chunk')
        with open(savefile_path, 'wb') as f:

            token_buffer = []
            for i, story in enumerate(iter_jsonl(loadfile_path)):
                # Read 1 story at a time from json and append the story tokens in the list
                token_buffer.extend(tokenize_tinystories(story['text'], enc))

                # Once batch size is reached, we convert te list into a np array and dump imn th binary file
                if (i+1)%chunk_size==0:
                    print(f'#Samples saved: {i+1}')
                    token_buffer_np = np.array(token_buffer, dtype=np.uint16)
                    token_buffer_np.tofile(f)

                    token_buffer = []

            # Write the final partial batch        
            if token_buffer:
                token_buffer_np = np.array(token_buffer, dtype=np.uint16)
                token_buffer_np.tofile(f)
        print(f'Saving complete')

    # File size:
    savefile_size_mb = savefile_path.stat().st_size/(1024**2)
    print('Tokenizing TinyStories and saving binary data complete')
    print(f' File path: {savefile_path} | Size: {savefile_size_mb:2f} MB')

    