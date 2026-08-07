from datasets import load_dataset
from pathlib import Path
import json
from typing import Dict


def download_tinystories(savedir_path:str="data/tinystories", max_limit=None)->Dict:

    print('='*50)
    print('Downloading TinyStories from HF...')
    print('='*50)

    # Define path for tinystories data
    ts_raw_dir_path = Path(savedir_path)/"raw"

    # Create directories
    ts_raw_dir_path.mkdir(parents=True, exist_ok=True)

    # File name for text file
    ts_raw_train_file_path = ts_raw_dir_path/"train.jsonl"
    ts_raw_val_file_path = ts_raw_dir_path/"val.jsonl"

    # Check if file already exists
    if ts_raw_train_file_path.exists():
        print("JSON files exist - download skipped")
    else:
        # Download file
        print('Starting download')
        dataset = load_dataset("roneneldan/TinyStories")

        # Save Training + Validation  file
        for split in ['train', 'validation']:
            
            print(f'Saving .jsonl file for split: {split}')
            if split == "train":
                filepath = ts_raw_train_file_path
            else:
                filepath = ts_raw_val_file_path
            with open(filepath, 'w', encoding='utf-8') as f:
                for i, story in enumerate(dataset[split]):
                    record = {
                        'id':i,
                        "text":story['text']
                    }
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
                    # Track progress
                    if (i+1)%10000==0:
                        print(f'#Samples saved: {i+1}')
                    if max_limit is not None and i>=max_limit:
                        break
            print(f'Saving complete')


    # File sizes
    train_file_size_mb = ts_raw_train_file_path.stat().st_size/(1024**2)
    val_file_size_mb = ts_raw_val_file_path.stat().st_size/(1024**2)

    print('Downloading TinyStories data complete')
    print(f' Train file path: {ts_raw_train_file_path} | Size: {train_file_size_mb:2f} MB')
    print(f' Val file path: {ts_raw_val_file_path} | Size: {val_file_size_mb:2f} MB')

    return {
        "train": ts_raw_train_file_path,
        "val": ts_raw_val_file_path
    }
            
