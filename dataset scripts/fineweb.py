import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # Reduced to prevent memory issues

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
eot = tokenizer.encoder["<|endoftext|>"]

def tokenize(doc):
    tokens = [eot]
    tokens.extend(tokenizer.encode(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token values are out of bounds"
    
    return tokens_np

def write_datafile(filename, tokens_np):
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())

if __name__ == '__main__':
    nprocs = max(1, os.cpu_count() // 2)
    print(f"Using {nprocs} processes")

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=32):  # Adjust chunksize
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index}.npy")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar.close()  # Reset progress bar
                progress_bar = None
                all_tokens_np[:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count > 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index}.npy")
            write_datafile(filename, all_tokens_np[:token_count])