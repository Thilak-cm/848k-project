import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
from tqdm import tqdm
import time
import json
# set seed for reproducibility
torch.manual_seed(1337)

# initialize wandb
wandb.init(project="GPT 2 848K")
wandb.run.tags = ['shakespeare text generation', 'test run', 'gelu activation', 'scaled down model']

# pull from local folder
filename = 'tinyshakespeare.txt'
with open(filename, 'r') as f:
    text = f.read()

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

wandb.config.update(params)

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

class RotaryPositionEmbeddings(nn.Module):
    '''Rotary Position Embeddings, as described in the RoPE paper'''
    def __init__(self, config=params, base=10_000):
        super().__init__()
        self.base = base
        self.dim = n_emb # config.n_emb
        self.max_seq_len = block_size # config.block_size
        self.config = config
        self.rope_init()
    
    def rope_init(self):
        '''
        Initialize the RoPE cache with sin and cos values for each position.
        '''
        # Compute thetas for sin and cos
        theta = torch.pow(self.base, -2 * torch.arange(0, self.dim // 2).float() / self.dim)
        self.register_buffer('theta', theta, persistent=False)
        self.build_rope_cache()
    
    def build_rope_cache(self):
        '''
        Build the RoPE cache for the given block size.
        '''
        seq_idx = torch.arange(self.max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

        # Compute position * theta for sin and cos
        # idx_theta = seq_idx.view(-1, 1) * self.theta.view(1, -1) # same functionality as einsum
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        # hs_half = self.config.n_embed // self.config.n_head // 2  # Calculate hs // 2
        hs_half = n_emb // n_heads // 2  # Calculate hs // 2
        idx_theta = idx_theta[:, :hs_half]  # Slice to match hs_half

        # Precompute sin and cos
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)  # Shape: [block_size, n_emb // 2, 2]
        self.register_buffer('cache', cache, persistent=False)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): Input tensor of shape [b, seq_len, nh, hs].
        
        Returns:
            torch.Tensor: Rotated tensor of the same shape as input.
        '''
        b, seq_len, nh, hs = x.shape  # Extract input dimensions
        # TODO: if n_emb is 32, nh is 2, hs should be 16 (n_emb // nh) but it is 8, why?
        print(f"shape of x before reshaping in RoPE: {x.shape}")

        # Slice the RoPE cache to match the sequence length
        print(f"shape of cache before slicing in RoPE: {self.cache.shape}")
        # rope_cache = self.cache[:seq_len].to(x.device)  # Shape: [seq_len, n_emb // 2, 2]
        rope_cache = self.cache[:seq_len, :hs // 2].to(x.device)
        # Reshape input for rotation (split last dim into pairs)
        x = x.reshape(*x.shape[:-1], -1, 2)  # Shape: [b, seq_len, nh, hs // 2, 2]
        print(f"shape of x after reshaping in RoPE: {x.shape}")
        # this or x = x.view(b, seq_len, nh, hs // 2, 2) # Shape: [b, seq_len, nh, hs // 2, 2]
        # Add singleton dimensions to rope_cache for broadcasting
        rope_cache = rope_cache.unsqueeze(0).unsqueeze(2)  # Shape: [1, seq_len, 1, h_s // 2, 2]
        # rope_cache = rope_cache.view(-1, x.size(1), 1, x.size(3), 2)
        print(f"Shape of rope_cache in RoPE: {rope_cache.shape}")
        
        # Perform the RoPE rotation
        rotated = torch.stack([
            x[..., 0] * rope_cache[..., 0] - x[..., 1] * rope_cache[..., 1],  # cos * even - sin * odd
            x[..., 1] * rope_cache[..., 0] + x[..., 0] * rope_cache[..., 1]   # sin * even + cos * odd
        ], dim=-1)  # Shape: [b, seq_len, nh, hs // 2, 2]

        # Flatten the last two dimensions back into the original shape
        print(f"shape of rotated before flattening in RoPE: {rotated.shape}")
        print(f"shape of rotated after flattening in RoPE: {rotated.flatten(-2).shape}")
        return rotated.flatten(-2).type_as(x)  # Shape: [b, seq_len, nh, hs]

class AttentionHead(nn.Module):
    '''one head of self-attention'''

    def __init__(self, head_size):
        super().__init__()
        # usually bias is not used in self-attention TODO: understand better why
        self.key = nn.Linear(n_emb, n_emb, bias=False)
        self.query = nn.Linear(n_emb, n_emb, bias=False)
        self.value = nn.Linear(n_emb, n_emb, bias=False)
        # triangular mask to prevent attending to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # using register buffer ensures that tril is not initialized as a param, so it won't be optimized during training
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbeddings()

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # BxTxC
        q = self.query(x) # BxTxC
        v = self.value(x) # BxTxC
        
        head_size = C // n_heads
        k = k.view(B, T, n_heads, head_size) # B x T x n_h x h_s
        q = q.view(B, T, n_heads, head_size) # B x T x n_h x h_s
        # apply rotary position embeddings
        k = self.rope(k)
        q = self.rope(q)

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
            # nn.ReLU(),
            nn.GELU(),
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
        # pos_emb = self.positional_embedding_table(torch.arange(T)) # time x channel
        x = token_emb # + pos_emb  # add positional embedding to token embedding
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
            idx = torch.cat([idx, next_token], dim=-1) # Bx(T+1)
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
required_tokens = total_params * 20
print(f"According to Chinchilla Scaling Law, you need at least {required_tokens:,} tokens to train this model effectively.")

# Check if the dataset meets the recommended number of tokens
if total_tokens >= required_tokens:
    print("✅ The dataset meets or exceeds the recommended number of tokens for effective training.")
else:
    shortfall = required_tokens - total_tokens
    print("⚠️ The dataset does NOT meet the recommended number of tokens for effective training.")
    print(f"  You are short by {shortfall:,} tokens.")
    print("  Consider either increasing the dataset size or reducing the model's parameters for optimal training.")

# Track start time of training
start_time = time.time()
patience = 1000
patience_counter = 0
avg_train_losses = []
avg_val_losses = []

for iter in tqdm(range(epochs), desc="Training Epochs"):
    train_losses = []
    val_losses = []

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

    # Evaluate at intervals and log to WandB
    if iter % eval_iter == 0:
        average_epoch_train_loss = sum(train_losses) / len(train_losses)
        average_epoch_val_loss = sum(val_losses) / len(val_losses)
        avg_train_losses.append(average_epoch_train_loss)
        avg_val_losses.append(average_epoch_val_loss)
        
        print(f"Epoch {iter}: Average Train Loss: {average_epoch_train_loss:.4f}, Average Validation Loss: {average_epoch_val_loss:.4f}")
        
        # Log to WandB at interval
        wandb.log({
            'avg_train_loss': average_epoch_train_loss,
            'avg_val_loss': average_epoch_val_loss
        })

        # Check for early stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
        patience_counter = patience_counter + 1 if average_epoch_val_loss > average_epoch_train_loss else 0

# End of training
end_time = time.time()
train_time = end_time - start_time

# Generate text
print(100*'*')
print(f"Generated Text:")
idx = torch.zeros((1,1), dtype=torch.long)
generated_text = decode(model.generate(idx, max_new_tokens=2000)[0].tolist())
print(generated_text)
print(100*'*')
print(100*'*')

# Calculate final average losses
final_avg_train_loss = sum(avg_train_losses) / len(avg_train_losses)
final_avg_val_loss = sum(avg_val_losses) / len(avg_val_losses)
print(f"Final Average Train Loss: {final_avg_train_loss:.4f}")
print(f"Final Average Validation Loss: {final_avg_val_loss:.4f}")

# Plot train and validation losses
plt.figure()
plt.plot(range(0, len(avg_train_losses) * eval_iter, eval_iter), avg_train_losses, label='Train Loss')
plt.plot(range(0, len(avg_val_losses) * eval_iter, eval_iter), avg_val_losses, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
# Save plot to WandB
plt.savefig('train_val_loss.png')
wandb.save('train_val_loss.png')

plt.show()

wandb.log({
    'final_avg_train_loss': final_avg_train_loss,
    'final_avg_val_loss': final_avg_val_loss,
    'time_to_train': train_time,
    'total_params': total_params
})

print(f"Total time to train model up to {epochs} epochs: {train_time:.2f} seconds")
wandb.finish()