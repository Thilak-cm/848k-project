# To run this code for 8 GPUs, use the following command
# torchrun --standalone --nproc_per_node=8 gpt2_flash_sinusoidal.py

from GPT2FlashAttention import GPT, GPTConfig
from dataset.shakespeare_dataset import DataLoaderLite
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
from torch.distributed import init_process_group, destroy_process_group
import math
import time
import wandb
import os

# Initialize wandb to this project
wandb.init(project="GPT 2 848K")

wandb.run.tags = ["GPT2 Flash Attention", "Sinusoidal Pos Embedding", "Shakesphere Dataset"]
wandb.run.name = "Shakesphere GPT2 Flash"


# This is for distributed data parallelism
ddp = int(os.environ.get('RANK', -1)) != -1

# If ddp is true, then we need to initialize the process group
if ddp:  # For legends
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])  # GPU 0 has rank 0, GPU 1 has rank 1, etc.
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # Local rank of GPU within the node
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

print("DDP: ", ddp)

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