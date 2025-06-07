# app.py

import streamlit as st
from transformers import pipeline

st.title("Two AI Models Talking")

@st.cache_resource(show_spinner=True)
def load_model(model_name):
    return pipeline("text-generation", model=model_name)

# Load two different models (you can pick any two, here both GPT-2 for example)
model1 = load_model("gpt2")
model2 = load_model("distilgpt2")

# Function to generate response given prompt and model
def generate_response(model, prompt):
    result = model(prompt, max_length=100, num_return_sequences=1)
    return result[0]['generated_text']

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    # Start conversation with a greeting from model1
    st.session_state.chat_history = [("Model 1", "Hello! How are you today?")]

st.write("### Conversation between Model 1 and Model 2:")

# Display chat history
for speaker, text in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {text}")

# Button to generate next turn
if st.button("Generate next turn"):
    last_speaker, last_text = st.session_state.chat_history[-1]

    if last_speaker == "Model 1":
        # Model 2 responds
        prompt = last_text + "\nModel 2:"
        response = generate_response(model2, prompt)
        # Clean response: remove prompt part
        response = response[len(prompt):].strip()
        st.session_state.chat_history.append(("Model 2", response))
    else:
        # Model 1 responds
        prompt = last_text + "\nModel 1:"
        response = generate_response(model1, prompt)
        response = response[len(prompt):].strip()
        st.session_state.chat_history.append(("Model 1", response))

st.write("---")
st.write("Click **Generate next turn** to continue the conversation.")
