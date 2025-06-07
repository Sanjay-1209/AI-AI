import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load model and tokenizer only once
@st.cache_resource
def load_model():
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# UI Settings
st.set_page_config(page_title="🤖 AI vs AI Chat", layout="wide")
st.title("🤖 AI vs AI: Let the Agents Talk")

# Sidebar Inputs
st.sidebar.header("Conversation Settings")
num_turns = st.sidebar.slider("Number of turns", 1, 10, 5)
max_new_tokens = st.sidebar.slider("Max length per response", 20, 100, 60)
starting_msg = st.sidebar.text_input("Starting message", "What do you think about the India-Pakistan rivalry?")

# Conversation Container
st.subheader("🗣️ Conversation Between Agents")
conversation = []

# Initialize message
current_input = starting_msg

for i in range(num_turns):
    # Agent A speaks
    res_a = generator(current_input, max_new_tokens=max_new_tokens, do_sample=True)[0]['generated_text']
    res_a_clean = res_a[len(current_input):].strip()
    conversation.append(("🧠 Agent A", res_a_clean))

    # Agent B replies
    res_b = generator(res_a, max_new_tokens=max_new_tokens, do_sample=True)[0]['generated_text']
    res_b_clean = res_b[len(res_a):].strip()
    conversation.append(("🤖 Agent B", res_b_clean))

    current_input = res_b  # Feed last reply to next agent

# Display Conversation
for speaker, msg in conversation:
    st.markdown(f"**{speaker}:** {msg}")

# Transcript section
with st.expander("📜 Full Conversation Transcript"):
    for speaker, msg in conversation:
        st.markdown(f"**{speaker}**: {msg}")
