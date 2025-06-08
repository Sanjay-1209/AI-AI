# import streamlit as st
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# st.set_page_config(page_title="2 AI Bots Chat", layout="wide")
# st.title("🤖 Two Chat AI Models Talking to Each Other")

# @st.cache_resource(show_spinner="Loading Zephyr model...")
# def load_chatbot():
#     model_name = "HuggingFaceH4/zephyr-7b-beta"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
#     pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
#     return pipe, tokenizer

# pipe, tokenizer = load_chatbot()

# def chat_response(pipe, prompt):
#     response = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]['generated_text']
#     # Strip the prompt part to get only the new reply
#     return response.replace(prompt, "").strip()

# # Initialize chat
# if "chat" not in st.session_state:
#     st.session_state.chat = [("AI 1", "Hey! How are you today?")]

# # Display conversation
# st.write("### 💬 Conversation History")
# for speaker, message in st.session_state.chat:
#     st.markdown(f"**{speaker}:** {message}")

# # Generate next turn
# if st.button("🗣️ Generate Next Turn"):
#     last_speaker, last_message = st.session_state.chat[-1]
#     if last_speaker == "AI 1":
#         prompt = f"<|system|>You are AI 2, having a smart conversation with AI 1.\n<|user|>{last_message}\n<|assistant|>"
#         response = chat_response(pipe, prompt)
#         st.session_state.chat.append(("AI 2", response))
#     else:
#         prompt = f"<|system|>You are AI 1, having a smart conversation with AI 2.\n<|user|>{last_message}\n<|assistant|>"
#         response = chat_response(pipe, prompt)
#         st.session_state.chat.append(("AI 1", response))

# st.info("Click **Generate Next Turn** to continue the AI-to-AI conversation.")





import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time

st.set_page_config(page_title="🤖 Two AIs Talking", layout="wide")
st.title("🤖 Two Real AI Models Talking to Each Other")

@st.cache_resource(show_spinner="Loading AI models...")
def load_model():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

pipe1 = load_model()  # AI Model 1
pipe2 = load_model()  # AI Model 2

# Session State
if "chat" not in st.session_state:
    st.session_state.chat = [("AI 1", "Hey AI 2, what are your thoughts on artificial intelligence?")]
if "running" not in st.session_state:
    st.session_state.running = False
if "turn" not in st.session_state:
    st.session_state.turn = "AI 2"

def generate_response(pipe, prompt):
    result = pipe(prompt, max_new_tokens=200, temperature=0.7, do_sample=True)[0]['generated_text']
    return result.replace(prompt, "").strip()

def continue_conversation():
    last_speaker, last_message = st.session_state.chat[-1]
    if last_speaker == "AI 1":
        prompt = f"<|system|>You are AI 2, having an intelligent discussion with AI 1.\n<|user|>{last_message}\n<|assistant|>"
        response = generate_response(pipe2, prompt)
        st.session_state.chat.append(("AI 2", response))
    else:
        prompt = f"<|system|>You are AI 1, having an intelligent discussion with AI 2.\n<|user|>{last_message}\n<|assistant|>"
        response = generate_response(pipe1, prompt)
        st.session_state.chat.append(("AI 1", response))

# Start/Stop Buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Start Conversation"):
        st.session_state.running = True
with col2:
    if st.button("⏹️ Stop Conversation"):
        st.session_state.running = False

# Display conversation
st.markdown("### 🧠 AI-to-AI Conversation:")
for speaker, message in st.session_state.chat[-10:]:
    st.markdown(f"**{speaker}:** {message}")

# Automatically run next turn every few seconds
if st.session_state.running:
    time.sleep(3)  # Wait before generating next response
    continue_conversation()
    st.experimental_rerun()
