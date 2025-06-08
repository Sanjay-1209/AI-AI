import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

st.set_page_config(page_title="2 AI Bots Chat", layout="wide")
st.title("🤖 Two Chat AI Models Talking to Each Other")

@st.cache_resource(show_spinner="Loading Zephyr model...")
def load_chatbot():
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe, tokenizer

pipe, tokenizer = load_chatbot()

def chat_response(pipe, prompt):
    response = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]['generated_text']
    # Strip the prompt part to get only the new reply
    return response.replace(prompt, "").strip()

# Initialize chat
if "chat" not in st.session_state:
    st.session_state.chat = [("AI 1", "Hey! How are you today?")]

# Display conversation
st.write("### 💬 Conversation History")
for speaker, message in st.session_state.chat:
    st.markdown(f"**{speaker}:** {message}")

# Generate next turn
if st.button("🗣️ Generate Next Turn"):
    last_speaker, last_message = st.session_state.chat[-1]
    if last_speaker == "AI 1":
        prompt = f"<|system|>You are AI 2, having a smart conversation with AI 1.\n<|user|>{last_message}\n<|assistant|>"
        response = chat_response(pipe, prompt)
        st.session_state.chat.append(("AI 2", response))
    else:
        prompt = f"<|system|>You are AI 1, having a smart conversation with AI 2.\n<|user|>{last_message}\n<|assistant|>"
        response = chat_response(pipe, prompt)
        st.session_state.chat.append(("AI 1", response))

st.info("Click **Generate Next Turn** to continue the AI-to-AI conversation.")
