import streamlit as st
import torch
import os
from tiktoken import get_encoding
from chat_with_model import GPTConfig, generate_response

# Import from scripts
from Model_Architectures.sinusoidal_arch import sinusoidal_GPT
from Model_Architectures.alibi_arch import alibi_GPT
from Model_Architectures.rope_arch import rope_GPT
from Model_Architectures.learnedPE_arch import learned_pe_GPT
from Model_Architectures.fire_arch import fire_GPT
from Model_Architectures.kerple_arch import kerple_GPT

# Title
st.title("Our 848K project: GPT-2 Unveiled: Comparative Insights")

# Sidebar for model selection
st.sidebar.header("Select a Model")
model_option = st.sidebar.selectbox(
    "Choose a Positional Encoding:",
    [
        "ALIBI",
        "FIRE",
        "Kerple",
        "Learned PE",
        "RoPE",
        "Sinusoidal",
    ]
)

# Map models to their respective architectures and paths
model_mapping = {
    "ALIBI": (alibi_GPT, "saved final models/final_alibi_model.pth"),
    "FIRE": (fire_GPT, "saved final models/final_fire_model.pth"),
    "Kerple": (kerple_GPT, "saved final models/final_kerple_model.pth"),
    "Learned PE": (learned_pe_GPT, "saved final models/final_learned_pe_model.pth"),
    "RoPE": (rope_GPT, "saved final models/final_rope_model.pth"),
    "Sinusoidal": (sinusoidal_GPT, "saved final models/final_sinusoidal_model.pth"),
}

# Load model based on user selection
if model_option:
    model_class, model_path = model_mapping[model_option]
    st.sidebar.write(f"Selected Model: {model_option}")

    # Load tokenizer and model
    st.write("Loading model...")
    tokenizer = get_encoding("gpt2")
    config = GPTConfig(vocab_size=50304)  # Match your trained model config
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Initialize and load the model
    model = model_class(config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace("_orig_mod.", "").replace("module._orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    st.write("Model loaded successfully!")

# Ensure chat history is properly initialized
if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
    st.session_state.chat_history = []

st.subheader("Chat")
user_input = st.text_input("Enter your message:")

if user_input and model_option:
    with st.spinner("Generating response..."):
        # Append user input as a separate entry
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Combine chat history into a clean model input
        full_input = "\n".join(
            [f"User: {m['content']}" if m['role'] == "user" else f"Model: {m['content']}"
             for m in st.session_state.chat_history]
        )

        # Generate response
        response, generation_time = generate_response(model, tokenizer, full_input)

        st.write(f"Response generated in {generation_time:.2f} seconds")

        # Append model response as a separate entry
        st.session_state.chat_history.append({"role": "model", "content": response})

        # Display chat history in alternating format
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Model:** {message['content']}")