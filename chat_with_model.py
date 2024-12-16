import torch
import torch.nn.functional as F
from tiktoken import get_encoding
from dataclasses import dataclass
import torch
import time

# Import model architectures
from Model_Architectures.sinusoidal_arch import sinusoidal_GPT
from Model_Architectures.alibi_arch import alibi_GPT
from Model_Architectures.rope_arch import rope_GPT
from Model_Architectures.learnedPE_arch import learned_pe_GPT
from Model_Architectures.fire_arch import fire_GPT
from Model_Architectures.kerple_arch import kerple_GPT

device = "mps" if torch.backends.mps.is_available() else "cpu"

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

# Main Script
if __name__ == "__main__":

    # Model options
    model_options = {
        "1": ("ALIBI", alibi_GPT, "saved final models/final_alibi_model.pth"),
        "2": ("FIRE", fire_GPT, "saved final models/final_fire_model.pth"),
        "3": ("Kerple", kerple_GPT, "saved final models/final_kerple_model.pth"),
        "4": ("Learned PE", learned_pe_GPT, "saved final models/final_learned_pe_model.pth"),
        "5": ("RoPE", rope_GPT, "saved final models/final_rope_model.pth"),
        "6": ("Sinusoidal PE", sinusoidal_GPT, "saved final models/final_sinusoidal_model.pth")
    }

    # Ask user to select a model
    print("Select a model to chat with:")
    for key, (name, _, _) in model_options.items():
        print(f"{key}: {name}")
    
    user_choice = None
    while user_choice not in model_options:
        user_choice = input("Enter the number of the model: ").strip()
        if user_choice not in model_options:
            print("Invalid choice. Please try again.")

    # Load the selected model
    model_name, model_class, model_path = model_options[user_choice]
    print(f"Loading {model_name}...")

    # Load model configuration and initialize the model
    config = GPTConfig(vocab_size=50304)
    model = model_class(config).to(device)

    # Load the pretrained model weights
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace("_orig_mod.", "").replace("module._orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # Load tokenizer
    tokenizer = get_encoding("gpt2")

    print("Chat with the model! Type 'exit' to quit.")
    while True:
        print(50 * "-")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = generate_response(model, tokenizer, user_input)
        print(f"{model_name} Model: {response}")