import re
import base64
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Set Streamlit page configuration (optional)
st.set_page_config(page_title="Groq Streaming Chat", layout="centered")

def process_thinking_stream(stream, response_placeholder, thinking_placeholder):
    """Process streaming response with real-time display."""
    response_content = ""
    thinking_content = ""
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response_content += chunk.choices[0].delta.content
            response_placeholder.markdown(response_content)
        if hasattr(chunk.choices[0].delta, 'reasoning') and chunk.choices[0].delta.reasoning:
            thinking_content += chunk.choices[0].delta.reasoning
            thinking_placeholder.markdown(thinking_content)
    
    return thinking_content, response_content

def display_message(message):
    """Display a single message in the chat interface."""
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        if role == "assistant":
            thinking_content = message.get("thinking")
            display_assistant_message(message["content"], thinking_content)
        else:
            st.markdown(message["content"])

def display_assistant_message(content, thinking_content=None):
    """Display assistant message with thinking content if present."""
    # Display thinking content in expander if present
    if thinking_content and thinking_content.strip():
        with st.expander("🧠 Thinking process", expanded=False):
            st.markdown(thinking_content)
    
    # Display response content in the main chat area
    if content:
        st.markdown(content)

def display_chat_history():
    """Display all previous messages in the chat history."""
    for message in st.session_state["messages"]:
        if message["role"] != "system":  # Skip system messages
            display_message(message)

def clean_messages_for_api(messages):
    """Remove thinking field from messages for API call."""
    return [{k: v for k, v in msg.items() if k != "thinking"} for msg in messages]

@st.cache_resource
def get_chat_model():
    """Get a cached instance of the chat model."""
    client = Groq()
    return lambda messages: client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=clean_messages_for_api(messages),
        stream=True,
        include_reasoning=True,
        reasoning_effort="high",
    )

def handle_user_input():
    """Handle new user input and generate assistant response."""
    if user_input := st.chat_input("Type your message here..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            # Create placeholders for streaming content
            thinking_expander = st.expander("🧠 Thinking process", expanded=False)
            thinking_placeholder = thinking_expander.empty()
            response_placeholder = st.empty()
            
            chat_model = get_chat_model()
            stream = chat_model(st.session_state["messages"])
            
            thinking_content, response_content = process_thinking_stream(
                stream, response_placeholder, thinking_placeholder
            )
            
            # Save the complete response
            st.session_state["messages"].append(
                {"role": "assistant", "content": response_content, "thinking": thinking_content}
            )

def main():
    """Main function to handle the chat interface and streaming responses."""
    # Load and encode logos
    openai_logo = base64.b64encode(open("assets/openai.png", "rb").read()).decode()
    groq_logo = base64.b64encode(open("assets/groq.png", "rb").read()).decode()
    
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 1rem;'>
            <img src="data:image/png;base64,{openai_logo}" width="40" style="vertical-align: middle; margin-right: 10px;">
            GPT-OSS Chat
            <img src="data:image/png;base64,{groq_logo}" width="40" style="vertical-align: middle; margin-left: 10px;">
        </h1>
        <h4 style='color: #666; margin-top: 0;'>With thinking UI! 💡</h4>
    </div>
    """, unsafe_allow_html=True)
    
    display_chat_history()
    handle_user_input()

if __name__ == "__main__":
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    main()
