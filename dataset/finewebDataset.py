import requests
import torch
import tiktoken
import numpy as np
import os

def load_tokens(filename):
    try: npt = np.load(filename, allow_pickle=True)
    except: npt = np.fromfile(filename, dtype=np.uint16)  # Replace dtype as needed

    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt 


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes,split, device ='cpu',):
        self.B, self.T = B, T
        self.device = device
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        master_process = process_rank ==0
        #get the shared filenames
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(CURRENT_DIR, "edu_fineweb10B")
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        
        #state, init and shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank


    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.tensor(self.tokens[self.current_position:self.current_position + B*T + 1])
        x = buf[:-1].view(B, T).to(self.device) #inputs
        y = buf[1:].view(B, T).to(self.device)  #targets
        
        # We need to advance position B*T*num_processes to get the next batch in tensor
        self.current_position += B*T*self.num_processes

        # If loading the next shard would be out of bounds, advance to the next shard
        if self.current_position +(B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x,y

