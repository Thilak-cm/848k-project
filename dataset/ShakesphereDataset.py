import requests
import torch
from transformers import AutoTokenizer



class DataLoaderLite:
    def __init__(self, B, T, device='cpu'):
        self.B, self.T = B, T
        self.device = device
        information = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        text = information.text
        # If you use this directly => tokens = tokenizer.encode(text, return_tensors='pt')
        # You'll get a warning because the text is too long and the model is too small because the model can take only 1024 tokens
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokens = tokenizer.encode(text, return_tensors='pt')
        print(f"Loaded {len(self.tokens[0])} tokens")
        print(f"1 epoch = {len(self.tokens[0]) // (self.B * self.T)} iterations")

        # State 
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = torch.tensor(self.tokens[0][self.current_position:self.current_position + B*T + 1])
        x = buf[:-1].view(B, T).to(self.device) 
        y = buf[1:].view(B, T).to(self.device)
        self.current_position += B*T

        if self.current_position + B*T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y

