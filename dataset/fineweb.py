import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)

# Set the cache directory
DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), local_dir)
os.make_dirs(DATA_CACHE_DIR, exist_ok=True)

# Download the dataset
fw = load_dataset("fineweb", name=remote_name, cache_dir=DATA_CACHE_DIR, train=True)

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
eot = tokenizer.encoder["<|endoftext|>"]
def tokenize(doc):
    tokens = [eot]
    tokens.extend(tokenizer.encode(doc["text"]))
    tokens_np = np.array(tokens)
    tokens = np.array(tokens, dtype=np.uint16)
    # Sanity check - tokens should be within bounds - 16 bit (for GPT2, we use 50k - here, it is 65k)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token values are out of bounds"
    return tokens

# Tokenize the dataset
def write_datafile(filename, tokens_np):
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())

# Tokenize all documents and write to output shards
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # if we have enough tokens to fill up the current shard, write it out
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))

        # otherwise, split the tokens between the current shard and the next shard
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index}.npy")
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:] = tokens[:remainder]
            # write out current shard
            write_datafile(f"edufineweb_{split}_{shard_index}.np", all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
        
    # write out the last shard
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index}.npy")
        write_datafile(filename, all_tokens_np[:token_count])
        shard_index += 1