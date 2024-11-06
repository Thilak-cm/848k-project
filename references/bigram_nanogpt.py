import urllib.request
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# pull from local folder
filename = 'tinyshakespeare.txt'
with open(filename, 'r') as f:
    text = f.read()

# get vocab
vocab = list(sorted(set(text)))
vocab_size = len(vocab)

# character level encoding and decoding
stoi = {c: i for i, c in enumerate(vocab)}
# itos = {i: c for i, c in enumerate(vocab)}
# alternate way of creating decoder func
itos = {i: c for c, i in stoi.items()}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])

# encode full dataset
data = torch.tensor(encode(text), dtype=torch.long)

# train test split, 85% split
train_size = int(0.85 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# create block sizes of 8
block_size = 8
epochs = 2000
eval_iter = 200
train_dataset = data[:block_size + 1]

torch.manual_seed(1337)
batch_size = 4 # how many sequences we will process in parallel, each of these sequences is block_size long
block_size = 8 # the length of each sequence

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token in the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both of shape (batch_size, block_size) aka (B, T)
        logits = self.token_embedding_table(idx) # Batch x time x channel
        if targets is None:
            loss = None
        else:
            # loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) # we could do this, but its hard to understand, so
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) 

        return logits, loss

    # auto regressive generation
    def generate(self, idx, max_new_tokens):
        # idx is BxT
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # pluck out last column in time dimension, because this is the generated predictions for what comes next
            logits = logits[:, -1, :] # keep only the last token for each sequence in the batch aka BxC
            probs = F.softmax(logits, dim=-1) # BxC
            # sample from the distribution
            next_tokens = torch.multinomial(probs, num_samples=1) # Bx1
            # append newly generated token to input idx to obtain new input for next generation iteration
            idx = torch.cat([idx, next_tokens], dim=1) # Bx(T+1)
        return idx

model = BigramLanguageModel(vocab_size)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
batch_size = 32

for iter in range(epochs):
    # evalute loss every eval_iter number of epochs to ensure smooth loss curve
    if iter % eval_iter == 0:
        averaged_loss = estimate_loss()
        print(f"Epoch: {iter}, train loss: {averaged_loss['train']}, val loss: {averaged_loss['val']}")
    
    # fetch batches
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = model(xb, yb)

    # set gradients to zero at start of every new epoch
    optimizer.zero_grad(set_to_none=True)

    # backprop
    loss.backward()

    # gradient update
    optimizer.step()

print(100*'*')
print(f"Generated Text:")
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))

