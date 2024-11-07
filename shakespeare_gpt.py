import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
from tqdm import tqdm
import time
import json
torch.manual_seed(1337)

# initialize wandb
wandb.init(project="GPT 2 848K")
wandb.run.tags = ['GPT 1', 'test run']

# pull from local folder
filename = 'tinyshakespeare.txt'
with open(filename, 'r') as f:
    text = f.read()

# TODO: count how many params you're using in this code, and implement chinchilla law to understand how much data you need to ensure you aren't under training
# get vocab
vocab = list(sorted(set(text)))
vocab_size = len(vocab)

scaled_up = False
if scaled_up:
    with open('gpt1_scaled_up_params.json', 'r') as f:
        params = json.load(f)
else:
    with open('gpt1_small_params.json', 'r') as f:
        params = json.load(f)

# model parameters
n_layer = params['n_layer']
n_heads = params['n_heads']
n_emb = params['n_emb']
block_size = params['block_size']
batch_size = params['batch_size']
learning_rate = params['learning_rate']
epochs = params['epochs']
eval_iter = params['eval_iter']
dropout = params['dropout']
train_test_split = params['train_test_split']

# Check if MPS (Metal Performance Shaders) is available for use on Mac
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print(f"Using device: {device}")

# character level encoding and decoding
stoi = {c: i for i, c in enumerate(vocab)}
# itos = {i: c for i, c in enumerate(vocab)}
# alternate way of creating decoder func
itos = {i: c for c, i in stoi.items()}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])

# encode full dataset
data = torch.tensor(encode(text), dtype=torch.long)

# train test split
train_size = int(train_test_split * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

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
        # usually bias is not used in self-attention TODO: understand better why
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        # triangular mask to prevent attending to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # using register buffer ensures that tril is not initialized as a param, so it won't be optimized during training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # BxTxC
        q = self.query(x) # BxTxC
        v = self.value(x) # BxTxC
        # compute attention scores
        # could potentially be optimized by using einsum? TODO: understand how
        # could potentially use lora's code to optimize this
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # BxTxC @ BxCxT (because of transposing second last and last dim of k) --> BxTxT
        # BxTxT: the TxT part of this attention matrix is where the quadratic complexity dependent on context length comes from
        # * C ** -0.5 is the one over root dk scaling factor in the attention formula
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # wherever tril is 0, in that position of wei, replace existing value with -inf
        # :T, :T is sliced to prevent index out of bounds error (for the case where block_size is not equal to T)
        wei = torch.softmax(wei, dim=-1) # TODO: understand why we softmax on the last dim
        wei = self.dropout(wei) # dropout on attention scores, randomly set some of them to 0
        # perform aggregation of values with attention scores
        out = wei @ v # BxTxT @ BxTxC --> BxTxC
        # out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # BxTxC
        # back to the dims we started with
        return out

class MultiHeadAttention(nn.Module):
    '''multi headed self attention'''

    def __init__(self, num_heads, head_size):
        super().__init__() # This initializes nn.Module (parent class from which MultiHeadAttention inherits from) before 
        # initializing anything in this child class
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
            nn.ReLU(), # TODO: use GELU instead of ReLU
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
        x = x + self.sa(self.ln1(x)) # residual connection # TODO: test using layer norm after sa and ffn as in original transformer paper 
        # and understand why there was an improvement in the new method
        x = x + self.ffn(self.ln2(x)) # residual connection (damn that was a very easy change to make)
        return x


class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token in the lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb) # W_E in GPT-2
        self.positional_embedding_table = nn.Embedding(block_size, n_emb) # W_P in GPT-2
        self.blocks = nn.Sequential(*[Block(n_emb, num_heads=n_heads) for _ in range(n_layer)]) # 4 blocks as per GPT-2 
        # asterisk is used here to unpack the list of blocks so it can be passed as individual elements to nn.Sequential and not as one big list
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

model = NanoGPT()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # TODO: try adding a lr schedule

# Calculate and display the total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params:,}")

# Calculate and display the total number of tokens in the dataset
total_tokens = len(data)
print(f"Total number of tokens in the dataset: {total_tokens:,}")

# Chinchilla Scaling Law suggests that the optimal number of tokens should be about 2 times the number of parameters.
# According to Chinchilla Law, we need at least 2 * total_params tokens
required_tokens = total_params * 2
print(f"According to Chinchilla Scaling Law, you need at least {required_tokens:,} tokens to train this model effectively.")

# Check if the dataset meets the recommended number of tokens
if total_tokens >= required_tokens:
    print("✅ The dataset meets or exceeds the recommended number of tokens for effective training.")
else:
    shortfall = required_tokens - total_tokens
    print("⚠️ The dataset does NOT meet the recommended number of tokens for effective training.")
    print(f"  You are short by {shortfall:,} tokens.")
    print("  Consider either increasing the dataset size or reducing the model's parameters for optimal training.")

# Track best losses and store losses for plotting
best_train_loss = float('inf')
best_val_loss = float('inf')
train_losses = []
val_losses = []

# Training loop with early stopping
start_time = time.time()
patience = 1000
patience_counter = 0

for iter in tqdm(range(epochs), desc="Training Epochs"):
    # Training phase
    model.train()  # Set model to training mode
    xb, yb = get_batch('train')
    logits, train_loss = model(xb, yb)

    # Zero gradients, backward pass, and optimizer step
    optimizer.zero_grad(set_to_none=True)
    train_loss.backward()
    optimizer.step()
    train_losses.append(train_loss.item())

    # Validation phase
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        X_val, Y_val = get_batch('val')
        logits, val_loss = model(X_val, Y_val)
    val_losses.append(val_loss.item())

    # Log and print average train and validation losses
    if iter % eval_iter == 0:
        print(f"Epoch: {iter}, Train Loss: {train_loss.item()}, Val Loss: {val_loss}")
    wandb.log({
        'train_loss': train_loss.item(),
        'val_loss': val_loss
    })

    # Track best losses
    if train_loss.item() < best_train_loss:
        best_train_loss = train_loss.item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset patience counter if validation loss improves
    else:
        patience_counter += 1  # Increment patience counter if validation loss does not improve

    # # Check for early stopping
    # if patience_counter >= patience:
    #     print(f"Early stopping triggered after {iter} epochs.")
    #     break

end_time = time.time()
train_time = end_time - start_time

print(100*'*')

print(f"Generated Text:")
idx = torch.zeros((1,1), dtype=torch.long)
generated_text = decode(model.generate(idx, max_new_tokens=2000)[0].tolist())
print(generated_text)
print(100*'*')
print(100*'*')

plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='val loss')
plt.legend()
plt.show()
# save plot to wandb
plt.savefig('train_val_loss.png')
wandb.save('train_val_loss.png')

print(f"Best Train Loss: {best_train_loss}")
print(f"Best Validation Loss: {best_val_loss}")

# Ensure train_time and other parameters are defined before logging
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
    'train_test_split': train_test_split,
    'total_params': total_params
})

print(f"Total time to train model up to {epochs} epochs: {train_time:.2f} seconds")
wandb.finish()