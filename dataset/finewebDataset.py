import requests
import torch
from transformers import AutoTokenizer



class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, device='cpu'):
        self.B, self.T = B, T
        self.device = device
        self.process_rank = process_rank
        self.num_processes = num_processes
        information = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        text = information.text
        # If you use this directly => tokens = tokenizer.encode(text, return_tensors='pt')
        # You'll get a warning because the text is too long and the model is too small because the model can take only 1024 tokens
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokens = tokenizer.encode(text, return_tensors='pt')
        print(f"Loaded {len(self.tokens[0])} tokens")
        print(f"1 epoch = {len(self.tokens[0]) // (self.B * self.T)} iterations")

        # State 
        # Process 0 will start from 0, Process 1 will start from B*T, Process 2 will start from 2*B*T, etc.
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.tensor(self.tokens[0][self.current_position:self.current_position + B*T + 1])
        x = buf[:-1].view(B, T).to(self.device) 
        y = buf[1:].view(B, T).to(self.device)
        
        # We need to jump B*T*num_processes to get the next batch
        self.current_position += B*T*self.num_processes

        # If we reach the end of the dataset, we need to start from the beginning
        if self.current_position + B*T*self.num_processes + 1 >= len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y

