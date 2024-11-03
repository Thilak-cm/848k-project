import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
from tqdm import tqdm
import time
torch.manual_seed(1337)

wandb.init(project="bigram_nanogpt")
wandb.run.tags = ['attention heads', 'multi headed attention', 'residual connections', 'feed forward nn', 'added layer norm']
wandb.run.notes = 'nano gpt'

# pull from local folder
filename = 'tinyshakespeare.txt'
with open(filename, 'r') as f:
    text = f.read()

# get vocab
vocab = list(sorted(set(text)))
vocab_size = len(vocab)
n_emb = 384
learning_rate = 1e-4
# create block sizes of 8
block_size = 256
epochs = 5000
eval_iter = 200
n_layer = 6
n_heads = 6
# so each head will have 64 dimensions
dropout = 0.2 # 20% will be zeroed out
train_test_split = 0.85
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

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
train_size = int(train_test_split * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

torch.manual_seed(1337)
batch_size = 4 # how many sequences we will process in parallel, each of these sequences is block_size long
block_size = 8 # the length of each sequence

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class AttentionHead(nn.Module):
    '''one head of self-attention'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # BxTxC
        q = self.query(x) # BxTxC
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # BxTxC @ BxCxT (because of transposing second last and last dim of k) --> BxTxT
        # BxTxT: the TxT part of this attention matrix is where the quadratic complexity dependent on context length comes from
        # * C ** -0.5 is the one over root dk scaling factor in the attention formula
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # wherever tril is 0, in that position of wei, replace existing value with -inf
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei) # dropout on attention scores, randomly set some of them to 0
        v = self.value(x)
        # perform aggregation of values with attention scores
        out = wei @ v # BxTxT @ BxTxC --> BxTxC
        # back to the dims we started with
        return out

class MultiHeadAttention(nn.Module):
    '''multi headed self attention'''

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_emb, n_emb) # linear layer to project concatenated heads output back to n_emb
        # project back into the residual pathway
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # BxTxC
        out = self.projection(out)
        return self.dropout(out)

class FeedForwardNN(nn.Module):
    '''simple one layer linear nn'''

    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb), # add a factor of 4 to n_emb as per GPT-2, just to make it more expressive, increasing complexity and computation
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb), # linear projection back into the residual pathway
            nn.Dropout(dropout) # add right before connetion before residual connection
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    '''transformer block: create multiple blocks and concatenate them'''

    def __init__(self, n_emb, num_heads):
        super().__init__()
        head_size = n_emb // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffn = FeedForwardNN(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection
        x = x + self.ffn(self.ln2(x)) # residual connection (damn that was a very easy change to make)
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token in the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb) # W_E in GPT-2
        self.positional_embedding_table = nn.Embedding(block_size, n_emb) # W_P in GPT-2
        self.blocks = nn.Sequential(*[Block(n_emb, num_heads=n_heads) for _ in range(n_layer)]) # 4 blocks as per GPT-2 # TODO; understand the syntax of the * here
        # also this is just a simpler representation of the previous thing we did, where we had a list of blocks and we individually called them
        self.lm_head = nn.Linear(n_emb, vocab_size) # W_o in GPT-2

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both of shape (batch_size, block_size) aka (B, T)
        token_emb = self.token_embedding_table(idx) # Batch x time x channel (here channel is now n_emb)
        pos_emb = self.positional_embedding_table(torch.arange(T)) # time x channel
        x = token_emb + pos_emb  # add positional embedding to token embedding
        x = self.blocks(x)
        logits = self.lm_head(x) # B, T, vocab size

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
            # get the last block_size tokens of the idx
            idx_cond = idx[:, -block_size:] # BxT
            logits, loss = self(idx_cond)
            # pluck out last column in time dimension, because this is the generated predictions for what comes next
            logits = logits[:, -1, :] # keep only the last token for each sequence in the batch aka BxC
            probs = F.softmax(logits, dim=-1) # BxC
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1) # Bx1
            # append newly generated token to input idx to obtain new input for next generation iteration
            idx = torch.cat([idx, next_token], dim=-1) # Bx(T+1) # TODO: understand why this is dim=-1
        return idx

model = BigramLanguageModel()

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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses = []
best_train_loss = float('inf')
best_val_loss = float('inf')

start_time = time.time()

for iter in tqdm(range(epochs), desc="Training Epochs"):
    # evaluate loss every eval_iter number of epochs to ensure smooth loss curve
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
    losses.append(loss.item())

    wandb.log({'loss': loss.item()})

    if averaged_loss['train'] < best_train_loss:
        best_train_loss = averaged_loss['train']
    if averaged_loss['val'] < best_val_loss:
        best_val_loss = averaged_loss['val']

end_time = time.time()
train_time = end_time - start_time

print(100*'*')
print(f"Best Train Loss: {best_train_loss}")
print(f"Best Validation Loss: {best_val_loss}")
print(f"Generated Text:")
idx = torch.zeros((1,1), dtype=torch.long)
# save text to file
with open('generated_shakespeare_text.txt', 'w') as f:
    f.write(decode(model.generate(idx, max_new_tokens=2000)[0].tolist()))

with open('generated_shakespeare_text.txt', 'r') as f:
    print(f.read())
print(100*'*')

# have wandb save the text file
wandb.save('generated_shakespeare_text.txt')

# plot loss curve
plt.plot(losses, label='train')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
# plt.show()

# log epoch, learning rate, block size, batch size, embedding size, optimizer, patience, device, vocab size to wandb
wandb.log({
    'epochs': epochs,
    "learning_rate": learning_rate,
    "block_size": block_size,
    "batch_size": batch_size,
    "embedding_size": n_emb,
    "optimizer": "AdamW",
    "device": device,
    "vocab_size": vocab_size,
    "best_train_loss": best_train_loss,
    "best_val_loss": best_val_loss,
    'Training Time': train_time, 
    'dropout': dropout,
    'n_layer': n_layer,
    'n_heads': n_heads,
    'train_test_split': train_test_split
})

print(f"Total time to train model up to {epochs} epochs: {train_time:.2f} seconds")
wandb.finish()
