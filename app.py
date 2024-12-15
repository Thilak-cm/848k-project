import streamlit as st
import torch
import os
from tiktoken import get_encoding
from chat_with_model import GPT, GPTConfig, generate_response

# Title
st.title("Chat with Fine-Tuned GPT-2 Models")

# Sidebar for model selection
st.sidebar.header("Select a Model")
model_option = st.sidebar.selectbox(
    "Choose PE + Attention Mechanism:",
    ["ALIBI + Flash Attention", "RoPE + Flash Attention"]
)

# Load model based on user selection
model_path = "saved final models/"
pth_files = os.listdir(model_path)
selected_model_file = None

# Match model file to selection
if model_option == "ALIBI + Flash Attention":
    selected_model_file = [f for f in pth_files if "Alibi" in f][0]
elif model_option == "FIRE + Flash Attention":
    selected_model_file = [f for f in pth_files if "FIRE" in f][0]
elif model_option == "Kerple + Flash Attention":
    selected_model_file = [f for f in pth_files if "Kerple" in f][0]
elif model_option == "Learned PE + Flash Attention":
    selected_model_file = [f for f in pth_files if "Learned PE" in f][0]
elif model_option == "RoPE + Flash Attention":
    selected_model_file = [f for f in pth_files if "ROPE" in f][0]
elif model_option == "Sinusoidal + Flash Attention":
    selected_model_file = [f for f in pth_files if "Sinusoidal" in f][0]

if selected_model_file:
    selected_model_path = os.path.join(model_path, selected_model_file)
    st.sidebar.write(f"Selected Model: {model_option}")
    st.sidebar.write(f"Model Path: {selected_model_path}")

    # Load the tokenizer and model
    st.write("Loading model...")
    tokenizer = get_encoding("gpt2")
    config = GPTConfig(vocab_size=50304)  # Match your trained model config
    model = GPT(config)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load state dictionary
    state_dict = torch.load(selected_model_path, map_location=device)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    st.write("Model loaded successfully!")

# Ensure chat history is properly initialized
if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
    st.session_state.chat_history = []

st.subheader("Chat")
user_input = st.text_input("Enter your message:")

if user_input and selected_model_file:
    with st.spinner("Generating response..."):
        # Append user input as a separate entry
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Combine chat history into a clean model input
        full_input = "\n".join(
            [f"User: {m['content']}" if m['role'] == "user" else f"Model: {m['content']}"
             for m in st.session_state.chat_history]
        )

        # Generate response
        response = generate_response(model, tokenizer, full_input)

        # Append model response as a separate entry
        st.session_state.chat_history.append({"role": "model", "content": response})

        # Display chat history in alternating format
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Model:** {message['content']}")