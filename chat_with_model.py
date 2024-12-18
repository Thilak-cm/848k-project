import torch
import os
import torch.nn.functional as F
from tiktoken import get_encoding
from dataclasses import dataclass
import torch
import time
from huggingface_hub import hf_hub_download

# Import model architectures
from model_architectures.sinusoidal_arch import sinusoidal_GPT
from model_architectures.alibi_arch import alibi_GPT
from model_architectures.rope_arch import rope_GPT
from model_architectures.learnedPE_arch import learned_pe_GPT
from model_architectures.fire_arch import fire_GPT
from model_architectures.kerple_arch import kerple_GPT

device = "mps" if torch.backends.mps.is_available() else "cpu"
repo_id = "thillsss/848k-models"

@dataclass
class GPTConfig:
    block_size: int = 1024 # Maximum sequence length
    vocab_size: int = 50257 # 50k "Byte Pair Encodings" (BPE) vocab size + 256 bytes tokens + 1 <|endoftoken|>
    # special end of sequence token delimits document boundaries and can start generation as well
    n_layer: int = 12 # Number of transformer blocks (how deep is the model)
    n_head: int = 12 # Number of heads in the multi-head attention (how wide is the model)
    n_embed: int = 768 # Embedding dimensionality

# Chat Functionality
def generate_response(model, tokenizer, input_text, max_length=50):
    start_time = time.time()
    
    tokens = tokenizer.encode(input_text)
    tokens = torch.tensor(tokens, dtype=torch.long).to(device)
    tokens = tokens.unsqueeze(0)
    model.eval()

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokens)[0]
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == tokenizer.eot_token:
                break

    response = tokenizer.decode(tokens[0].tolist())
    end_time = time.time()
    print(f"Response generated in {end_time - start_time:.2f} seconds")
    
    return response, end_time - start_time

#import os
import torch
import torch.nn.functional as F
from tiktoken import get_encoding
from dataclasses import dataclass
import time
from huggingface_hub import hf_hub_download

# Import model architectures
from model_architectures.sinusoidal_arch import sinusoidal_GPT
from model_architectures.alibi_arch import alibi_GPT
from model_architectures.rope_arch import rope_GPT
from model_architectures.learnedPE_arch import learned_pe_GPT
from model_architectures.fire_arch import fire_GPT
from model_architectures.kerple_arch import kerple_GPT

device = "mps" if torch.backends.mps.is_available() else "cpu"
repo_id = "thillsss/848k-models"  # Your Hugging Face Hub repository

@dataclass
class GPTConfig:
    block_size: int = 1024  # Maximum sequence length
    vocab_size: int = 50257
    n_layer: int = 12  # Transformer layers
    n_head: int = 12  # Attention heads
    n_embed: int = 768  # Embedding size

# Chat Functionality
def generate_response(model, tokenizer, input_text, max_length=50):
    start_time = time.time()
    tokens = tokenizer.encode(input_text)
    tokens = torch.tensor(tokens, dtype=torch.long).to(device).unsqueeze(0)
    model.eval()

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokens)[0]
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == tokenizer.eot_token:
                break

    response = tokenizer.decode(tokens[0].tolist())
    end_time = time.time()
    print(f"Response generated in {end_time - start_time:.2f} seconds")
    return response, end_time - start_time

# Function to get model path
def get_model_path(model_filename):
    local_path = os.path.join("saved_final_models", model_filename)
    if not os.path.exists(local_path):
        print(f"{model_filename} not found locally. Downloading from Hugging Face Hub...")
        return hf_hub_download(repo_id=repo_id, filename=model_filename)
    print(f"Loading {model_filename} from local directory.")
    return local_path

# Main Script
if __name__ == "__main__":
    # Model options
    model_mapping = {
        "ALIBI": (alibi_GPT, "final_alibi_model.pth"),
        "FIRE": (fire_GPT, "final_fire_model.pth"),
        "Kerple": (kerple_GPT, "final_kerple_model.pth"),
        "Learned PE": (learned_pe_GPT, "final_learned_pe_model.pth"),
        "RoPE": (rope_GPT, "final_rope_model.pth"),
        "Sinusoidal": (sinusoidal_GPT, "final_sinusoidal_model.pth"),
    }

    # Prompt user to select a model
    print("Select a model to chat with:")
    for i, key in enumerate(model_mapping.keys(), 1):
        print(f"{i}: {key}")

    user_choice = None
    options_list = list(model_mapping.keys())
    while user_choice not in map(str, range(1, len(options_list) + 1)):
        user_choice = input("Enter the number of the model: ").strip()
        if user_choice not in map(str, range(1, len(options_list) + 1)):
            print("Invalid choice. Please try again.")

    model_name = options_list[int(user_choice) - 1]
    model_class, model_filename = model_mapping[model_name]

    print(f"Loading {model_name} model...")

    # Get model file path (check local, download if not exists)
    model_file_path = get_model_path(model_filename)

    # Initialize and load the model
    config = GPTConfig(vocab_size=50304)
    model = model_class(config).to(device)
    state_dict = torch.load(model_file_path, map_location=device)
    state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # Load tokenizer
    tokenizer = get_encoding("gpt2")

    print("Chat with the model! Type 'exit' to quit.")
    while True:
        print(50 * "-")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response, _ = generate_response(model, tokenizer, user_input)
        print(f"{model_name} Model: {response}")
    # Load tokenizer__pycache__/
    tokenizer = get_encoding("gpt2")

    # sample prompt: i'm a language model. these are some of the things i can help you with:
    print("Chat with the model! Type 'exit' to quit.")
    while True:
        print(50 * "-")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response, _ = generate_response(model, tokenizer, user_input)
        print(f"{model_name} Model: {response}")