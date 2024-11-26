# To run this code for 8 GPUs, use the following command
# torchrun --standalone --nproc_per_node=8 gpt2.py

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
import numpy as np
from hellaswag import render_example, iterate_examples
import tiktoken
import torch._dynamo
torch._dynamo.config.suppress_errors = True

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, device ='cpu',):
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

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # UserWarning: To copy construct from a tensor, 
        # it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), 
        # rather than torch.tensor(sourceTensor)
        # buf = torch.tensor(self.tokens[self.current_position:self.current_position + B*T + 1])
        buf = self.tokens[self.current_position:self.current_position + B*T + 1].clone().detach()#.requires_grad_(True)
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

# pytorch can be serious about device vs device type so we need to set it correctly
# TODO: understand the difference between device and device type
device_type = "cuda" if device.startswith("cuda") else "cpu"

# This is for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

if master_process:
    # Initialize wandb to this project
    wandb.init(project="GPT 2 848K Nexus Cluster")

    wandb.run.tags = ["GPT2", "124M params", "10B tokens", "Kerple Attention", "Gelu", "Kerple Log Positional Encoding"]


def load_tokens(filename):
    try: npt = np.load(filename, allow_pickle=True)
    except: npt = np.fromfile(filename, dtype=np.uint16)  # Replace dtype as needed

    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt 


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

enc = tiktoken.get_encoding('gpt2')

# Good, bad and ugly numbers - Why 50304? 50304 % 128 = 0 and is even 
model = torch.load("/fs/class-projects/fall2024/cmsc848k/c848k017/Kerple/model_0.pth", map_location=device)

# count number of parameters
num_params = sum(p.numel() for p in model.parameters())
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if master_process:
    print(100 * '-')
    print(f"Total number of parameters: {num_params}, Trainable parameters: {num_trainable_params}")

total_tokens = 1e10 # 10B tokens

if master_process:
    # log parameters to wandb
    wandb.watch(model, log="all")

 
# This is for ddp
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
raw_model = model.module if ddp else model # Always contains the "raw" unwrapped model

if master_process:
    for name, param in raw_model.named_parameters():
        print(f"Layer: {name} | Number of parameters: {param.numel()}")


# This is for gradient accumulation
# total_batch_size = 2**19 # 500K tokens
B, T = 16, 1024


if master_process: # To print jsut one single time
    wandb.config.update({
    # Training parameters
    "batch_size": B,
    "sequence_length": T,
    #"total_batch_size": total_batch_size,
    "device": device,

    # Model parameters
    "embedding_size": raw_model.config.n_embed,
    "num_layers": raw_model.config.n_layer,
    "num_heads": raw_model.config.n_head,
    "vocab_size": raw_model.config.vocab_size,
    "dropout": 0,

    # Parameter counts
    "total_params": num_params,
    "trainable_params": num_trainable_params,
    })
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", device=device)

torch.cuda.empty_cache()
# This is for TF32 - 19 bits: 1 sign, 8 range and 10 mantissa
torch.set_float32_matmul_precision('high')
best_train_loss_accum = 1e9
avg_time = 0
avg_tokens_per_sec = 0



for epoch in range(0, 21000, 1000):
    if epoch < 20000:
        model = torch.load(f"/fs/class-projects/fall2024/cmsc848k/c848k017/Kerple/model_{epoch}.pth", map_location=device)
    else: 
        model = torch.load("/fs/class-projects/fall2024/cmsc848k/c848k017/Kerple/final_epoch_model.pth", map_location=device)
    
    # Python interpreter is very slow. So, we need to compile the model
    # If compiled, in GPU, instead of traversing from HBM to cache for each single operation, 
    # computation is done by traversing once  
    # This is for linux only
    model = torch.compile(model)
    # This is for ddp
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    raw_model = model.module if ddp else model # Always contains the "raw" unwrapped model


    model.eval()

    ########## once in a while evaluate our validation loss
    if master_process:  print("evaluating validation loss:")
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        print(f"validation loss: {val_loss_accum.item():.4f}")
        if master_process: 
            wandb.log({"val_loss": val_loss_accum.item()})
            # you might also want to add optimizer.state_dict() and
            # rng seeds etc., if you wanted to more exactly resume training


    #################### once in a while evaluate hellaswag
    if master_process: print('evaluating hellaswag benchmark performance')
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where epoch % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        # log the accuracy
        hellaswag_accuracy = acc_norm
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={hellaswag_accuracy:.4f}")
        if master_process: wandb.log({"hellaswag_accuracy": hellaswag_accuracy})


    ############## once in a while generate from the model (except epoch 0, which is noise)
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"rank {ddp_rank} sample {i}: {decoded}")


# Destroy all processes if ddp is true
if ddp: destroy_process_group()
