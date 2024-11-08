from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import time
import math
import inspect
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import os
from transformers import AutoTokenizer
import wandb

# Initialize wandb to this project
wandb.init(project="GPT 2 848K")

wandb.run.tags = ["GPT2 Flash Attention", "Sinusoidal Pos Embedding", "Shakesphere Dataset"]
wandb.run.name = "Shakesphere GPT2 Flash"

# GPT-2 is a decoder only transformer model
# %%
#This is for MLP block
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        # TODO: Instead of using tanh, we can use non approximate version also
        #There is not much difference between the time for tanh and real GELU
        self.gelu = nn.GELU(approximate='tanh') # gpt2 uses tanh approximation so we're using it
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed) 
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# This is for self attention block
class CausalSelfAttention(nn.Module):
    '''
    this is for self attention decoder block (cuz it's causal)
    '''
    def __init__(self, config):
        super().__init__() 
        assert config.n_embed % config.n_head == 0 # This is for checking if the number of heads is divisible by the embedding dimension
        
        # This is for query, key and value projections
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)  # combined linear projection for all three Q, K, V
        # This is for output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # This is for initializing the weights with 1.0
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # not really a bias but a mask, but following OpenAI naming convention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) 
        # reshaped like 1,1,T,T so that it can be broadcast with the batch size and attention head dimensions

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence Length, Embedding dimensionality (n_embed)
        # d_k = d_v = n_embed // n_head
        # n_head -> Number of heads in the multi-head attention
        # create query, key, value matrices
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)  # B x T x n_embed -> B x T x 3*n_embed -> q, k, v each of shape B x T x n_embed
        # hs (head size) = n_embed // n_head
        # C == n_emb == n_head * hs == n_head * d_k == n_head * d_v
        q = q.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # q (BxTxC) reshaped to B x T x n_head x hs then transposed to B x n_head x T x hs
        k = k.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # same for k
        v = v.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # same for v


        # Attention mechanism
        # att = (q @ k.transpose(-2, -1)) * (1.0 / ((k.size(-1)) ** 0.5))
        # # Masked Attention
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # wow who knew flash attention was so easy to implement
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * (self.n_embed // self.n_head))
        # Output projection
        y = self.c_proj(y)
        return y

# This is for transformer block
class Block(nn.Module):
    '''
    This is for transformer block
    '''

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # didn't realize it was this easy to implement residual connections
        x = x + self.attn(self.ln_1(x)) # clean residual connections are desrable for deep models form an optimization perspective
        x = x + self.mlp(self.ln_2(x)) # also we perform layer normalization before self attention and MLP, in contrast to the original transformer
        # this is because it is more stable to normalize the input to each sub-layer, rather than the output
        # this is called pre-normalization and is used in the "An Image is Worth 16x16 Words" paper
        return x

# default 124M randomly initialized model
@dataclass
class GPTConfig:
    block_size: int = 1024 # Maximum sequence length
    vocab_size: int = 50257 # 50k "Byte Pair Encodings" (BPE) vocab size + 256 bytes tokens + 1 <|endoftoken|>
    # special end of sequence token delimits document boundaries and can start generation as well
    n_layer: int = 12 # Number of transformer blocks (how deep is the model)
    n_head: int = 12 # Number of heads in the multi-head attention (how wide is the model)
    n_embed: int = 768 # Embedding dimensionality

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


        # Developing Transformer
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embed), # Token embedding weights
            'wpe': nn.Embedding(config.block_size, config.n_embed), # Positional embedding weights
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # All transformer blocks
            'ln_f': nn.LayerNorm(config.n_embed)
        })

        # Final Linear layer after all transformer blocks
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
    
        # Weight sharing scheme
        # This is for sharing the weights between token and positional embeddings
        # Reason: Since they are semantically similar, they should have similar weights
        self.lm_head.weight = self.transformer['wte'].weight

        # Initialize parameters with mean 0 and standard deviation 0.02 because 1/sqrt(768), 1/sqrt(1600)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # Here 2 is because one is for MLP and other is for attention mechanism
                # config.n_layer is number of transformer blocks
                # Why this? 
                # We send each layer after adding the residual connection and normalization
                # To make results' distribution normal with standard deviation = 0.02
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            # if module.padding_idx is not None:
            #     torch.nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, model block size is {self.config.block_size}"

        # IMP: Token and Positional Embeddings
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        pos_emb = self.transformer.wpe(pos)  #Positional Embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx)  #Token Embeddings of shape (B, T, n_embed)
        x = tok_emb + pos_emb # broadcast along the batch dimension

        # Forward pass through each transformer block
        for block in self.transformer.h:
            x = block(x)
        
        # Final Linear layer
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)

        # Loss function
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss


    @classmethod
    def from_pretrained(cls, model_type):
        ## This is for loading the pretrained model
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import AutoModelForCausalLM
        print("Loading weights from pretrained gpt:", model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768),           #124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),   #345M parameters
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),    #774M parameters
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600)        #1558M parameters
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('attn.bias')]

        # init a huggingface GPT2 model
        model_hf = AutoModelForCausalLM.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Check parameters match
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.bias')]
        assert set(sd_keys) == set(sd_keys_hf), f"Length mismatch - keys: {len(sd_keys)} and huggingface keys: {len(sd_keys_hf)}"

        # Copy shared weights
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys:
            if any(k.endswith(x) for x in transposed):
                assert sd[k].shape == sd_hf[k].shape[::-1]
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())  #This is transpose but will not give warnings
            else:
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, lr, device):
        # We need all the named parameters that require gradients
        param_dict = {k: v for k, v in self.named_parameters()}
        param_dict = {k: v for k, v in param_dict.items() if v.requires_grad}
        
        # Bias does not need weight decay 
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # Optimizer
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        # Return number of elements (numel), which is the number of parameters
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(f"# Decayed parameter tensors: {len(decay_params)} with {num_decay_params} parameters")
        print(f"# No Decay parameter tensors: {len(no_decay_params)} with {num_no_decay_params} parameters")

        # Check AdamW optimizer and use the fused verison if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using Fused AdamW: {fused_available}")
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay, fused=use_fused)
        return optimizer
#%%    
########################################################################################
# Comparison of the models
def compare(model, device):
    from transformers import AutoModelForCausalLM as A
    model_hf = A.from_pretrained('gpt2')
    model_hf.eval()
    model_hf.to(device)

    sd = model.state_dict()
    sd_hf = model_hf.state_dict()
    sd_keys = sd.keys()
    sd_keys_hf = sd_hf.keys()
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    for k in sd_keys:
        if not k.endswith('attn.bias'):
            if any(k.endswith(x) for x in transposed):
                assert sd[k].shape == sd_hf[k].shape[::-1]
                assert torch.allclose(sd[k], sd_hf[k].t(), atol=1e-5), f"Weight mismatch for key: {k}"
            
            else:
                assert sd[k].shape == sd_hf[k].shape
                assert torch.allclose(sd[k], sd_hf[k], atol=1e-5), f"Weight mismatch for key: {k}"
    print("All weights match")
########################################################################################



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


#%%

# This is for distributed data parallelism
ddp = int(os.environ.get('RANK', -1)) != -1

# If ddp is true, then we need to initialize the process group
if ddp:  # For legends
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])  # GPU 0 has rank 0, GPU 1 has rank 1, etc.
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # Local rank within the node
    ddp_world_size = int(os.environ['WORLD_SIZE']) # Number of GPUs

    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)   
    master_process = ddp_rank == 0  
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cpu' # For noobs
    if torch.cuda.is_available(): # For adults
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # For kids
        device = "mps"
    print(f"using device: {device}" )


# This is for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Good, bad and ugly numbers - Why 50304? 50304 % 128 = 0 and is even 
model = GPT(GPTConfig(vocab_size=50304)).to(device)
model.to(device)

# Python interpreter is very slow. So, we need to compile the model
# If compiled, in GPU, instead of traversing from HBM to cache for each single operation, 
# computation is done by traversing once  
# This is for linux only
# model = torch.compile(model)

# This is for ddp
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
raw_model = model.module if ddp else model # Always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr / 10
warmup_steps = 10
max_steps = 20
def get_lr(it):
    # 1) Linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) If it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) In between, use cosine learning rate decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_ratio))

# Now GPT-3 parameters are used for GPT-2
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # Optimizer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, lr=6e-4, device=device)


# This is for gradient accumulation
total_batch_size = 2**19 # 500K tokens
B, T = 4, 1024

#The below steps contain the number of steps to accumulate the gradients including multiple GPU steps too
assert total_batch_size % (B * T * ddp_world_size) == 0, f"Batch size {total_batch_size} is not divisible by B * T = {B * T}"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) 

if master_process: # To print jsut one single time
    print(f"Desired batch size: {total_batch_size}, Gradient Accumulation Steps: {grad_accum_steps}")
train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, device=device)

torch.cuda.empty_cache()
# This is for TF32 - 19 bits: 1 sign, 8 range and 10 mantissa
torch.set_float32_matmul_precision('high')
best_train_loss_accum = 1e9
avg_time = 0
avg_tokens_per_sec = 0
for i in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    # This is for gradient accumulation
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        #This is to use BP16
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, targets=y)
        
        # we have to scale loss to account for gradient accumulation
        # because the gradients just add on each successive backward()
        # addiion of gradients corresponds to a SUM in the objective, but 
        # instead of SUM, we want a MEAN. So we scale the loss by the number of gradient accumulation steps
        loss = loss / grad_accum_steps # This acts like normalizer since reduction is mean
        loss_accum += loss.item()
        
        # loss.backward() # Do the backward pass 
        if ddp: # Sync the gradients only on the last step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) 
        loss.backward() # Do the backward pass and synchronize the gradients.

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # Average the loss across all GPUs
    # Gradient global clipping: Why is this used? Because the gradients can be very large and can cause overflow
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Learning rate scheduler
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # This completes all the operations without starting new operation
    torch.cuda.synchronize()
    t1 = time.time()
    avg_time += t1 - t0
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0)
    avg_tokens_per_sec += tokens_per_sec

    best_train_loss_accum = min(best_train_loss_accum, loss_accum)
    # Wandb logging
    wandb.log({
        'epochs': i,
        "device": device,
        "learning_rate": get_lr(i - 1),
        "loss": loss_accum,
        "norm": norm,
        "mini_block_size": T,
        "mini_batch_size": B, 
        
        # Parameters for the model
        "embedding_size": model.config.n_embed,
        "num_layers": model.config.n_layer,
        "num_heads": model.config.n_head,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "vocab_size": model.config.vocab_size,
        "dropout": 0,
        
        #Optimizer and learning rate
        "optimizer": "AdamW",
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "max_lr": max_lr,
        "min_lr": min_lr,

        # Parameters for gradient accumulation
        "grad_accum_steps": grad_accum_steps,
        "total_batch_size": total_batch_size,
        "tokens_per_batch": train_loader.B * train_loader.T,
        "total_tokens_per_step": train_loader.B * train_loader.T * grad_accum_steps,

        # Parameters for time and number of tokens
        "current_epoch_time": t1 - t0,
        "tokens_per_sec": tokens_per_sec,
        "avg_time": avg_time / (i + 1),
        "avg_tokens_per_sec": avg_tokens_per_sec / (i + 1),
        "best_train_loss": best_train_loss_accum  # Best training loss
    })

    if master_process:
        print(f"Epoch: {i}, Loss: {loss_accum}, lr: {lr}, norm: {norm}, Time Difference: {(t1 - t0)* 1000}ms, #tokens/sec: {tokens_per_sec}")
# %%
print(f"Average time: {avg_time / max_steps * 1000}ms, Average tokens/sec: {avg_tokens_per_sec / max_steps}")


# Destroy all processes if ddp is true
if ddp: 
    destroy_process_group()


# # %%
# ########################################################################################
# # Loading a pretrained model and generating text
# # Generate text

# model = GPT.from_pretrained('gpt2')
# model.eval()
# model.to(device)
# compare(model, device)

# max_length = 30 # Maximum length of the generated text
# num_return_sequences = 5 # Number of different answers to generate

# # Prefix tokens
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('gpt2')

# tokens = tokenizer.encode("So, this morning I started studying for the interview in the lab. This was not", return_tensors='pt') # (8,)
# # tokens = torch.tensor(tokens, dtype=torch.long)
# # Repeat the tokens for the number of return sequences
# tokens = tokens.repeat(num_return_sequences, 1)  # (5, 8)
# x = tokens.to('cuda')

# # Another way
# # import tiktoken
# # enc = tiktoken.get_encoding('gpt2')
# # tokens = enc.encode("Hello, I'm a language model,")
# # tokens = torch.tensor(tokens, dtype=torch. long) # (8. )
# # tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# # x = tokens.to('cuda')


# # x - (B, T) - (Batch Size, Sequence Length)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     model.eval()
#     with torch.no_grad():
#         logits = model(x)[0] #x, loss
#         probs = F.softmax(logits[:, -1, :], dim=-1)
#         topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)  #k=50 is GPT-2 default
#         ix = torch.multinomial(topk_probs, num_samples=1)
#         xcol = torch.gather(topk_indices, -1, ix)
#         x = torch.cat((x, xcol), dim=1)

# #print generated text
# for i in range(num_return_sequences):
#     print(tokenizer.decode(x[i, :].tolist()))
#     # print(enc.decode(x[i, :].tolist()))
# ########################################################################################