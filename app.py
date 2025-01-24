import streamlit as st
import openai

# App title and configuration
st.set_page_config(page_title="Oregon Govt. Assistance Bot", layout="wide")
openai.api_key = st.secrets['OPEN_AI_API_TOKEN']


# Clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Title and clear button
st.title("Oregon Govt. Assistance Bot")
st.button("Clear Chat History", on_click=clear_chat_history)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate OpenAI response
def openAI_response(prompt_input):
    # Include conversation history for context
    messages = st.session_state.messages + [{"role": "user", "content": prompt_input}]
    
    response = openai.ChatCompletion.create(
        model="ft:gpt-4o-2024-08-06:dharani:fine-tune-final:AYiTKoBs",  # Update with your model ID
        messages=messages
    )
    message_content = response['choices'][0]['message']['content']
    return message_content

# User input handling
if prompt := st.chat_input(placeholder="Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = openAI_response(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
