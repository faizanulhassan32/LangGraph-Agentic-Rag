"""Streamlit chat interface for the Agentic RAG system."""

import asyncio
import os
import time
from typing import List, Dict

import streamlit as st
from langgraph_sdk import get_client

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        margin-bottom: 1rem;
        background-color: #fafafa;
    }
    .human-message {
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        padding: 12px;
        border-radius: 12px;
        margin: 10px 0;
        margin-left: 20px;
        color: #0d47a1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .ai-message {
        background-color: #e8f5e8;
        border: 1px solid #c8e6c8;
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        margin-right: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2d5016;
        font-weight: 500;
    }
    .sidebar-info {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "client" not in st.session_state:
    try:
        st.session_state.client = get_client(url="http://localhost:2024")
    except Exception as e:
        st.session_state.client = None
        st.error(f"Could not connect to LangGraph server: {e}")

# Sidebar configuration
with st.sidebar:
    st.header("🛠️ Configuration")
    
    # Model settings
    model = st.selectbox(
        "Model",
        options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
        index=0
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=1000,
        value=500,
        step=50
    )
    
    retrieval_threshold = st.slider(
        "Retrieval Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    st.markdown("---")
    
    # Information section
    st.markdown("""
    <div class="sidebar-info">
    <h4>📚 Document Sources</h4>
    <ul>
        <li>Reward Hacking in AI</li>
        <li>Hallucination in LLMs</li>
        <li>Diffusion Video Models</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Connection status
    if st.session_state.client:
        st.success("✅ Connected to LangGraph")
    else:
        st.error("❌ LangGraph server not connected")
        st.info("Run: `langgraph dev` to start the server")

# Main interface
st.markdown('<h1 class="main-header">🤖 Agentic RAG Chat Assistant</h1>', unsafe_allow_html=True)


# Chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    if st.session_state.messages:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="human-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👋 Start a conversation by asking a question below!")

# Input section
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question:",
            placeholder="e.g., What is reward hacking in AI systems?",
            label_visibility="collapsed"
        )
    
    with col2:
        submit_button = st.form_submit_button("Send 🚀", type="primary")

# Process user input
if submit_button and user_input and st.session_state.client:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show processing indicator
    with st.spinner("🤔 Agent is thinking..."):
        try:
            # Prepare configuration
            config = {
                "configurable": {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "retrieval_threshold": retrieval_threshold
                }
            }
            
            # Prepare input state
            input_state = {
                "question": user_input,
                "messages": [],
                "documents": [],
                "answer": "",
                "needs_retrieval": True,
                "retrieval_attempts": 0,
                "max_retrieval_attempts": 2
            }
            
            # Stream response from LangGraph
            response_placeholder = st.empty()
            current_response = ""
            
            async def get_response():
                try:
                    async for chunk in st.session_state.client.runs.stream(
                        None,
                        "agent",  # Assistant name
                        input=input_state,
                        config=config
                    ):
                        if hasattr(chunk, 'data') and chunk.data:
                            if isinstance(chunk.data, dict) and 'answer' in chunk.data:
                                if chunk.data['answer']:
                                    return chunk.data['answer']
                    return "I apologize, but I couldn't generate a response."
                except Exception as e:
                    return f"Error: {str(e)}"
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                assistant_response = loop.run_until_complete(get_response())
            finally:
                loop.close()
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"I apologize, but I encountered an error: {str(e)}"
            })
    
    # Rerun to update the chat display
    st.rerun()

elif submit_button and user_input and not st.session_state.client:
    st.error("Please make sure the LangGraph server is running (`langgraph dev`) and refresh the page.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Messages", len(st.session_state.messages))

with col2:
    if st.session_state.client:
        st.metric("Status", "🟢 Connected")
    else:
        st.metric("Status", "🔴 Disconnected")

with col3:
    st.metric("Model", model)

# Instructions at the bottom
with st.expander("ℹ️ How to use this chat"):
    st.markdown("""
    ### Getting Started:
    1. Make sure you have set your OpenAI API key: `export OPENAI_API_KEY="your-key"`
    2. Start the LangGraph server: `langgraph dev`
    3. Ask questions about the available topics
    
    ### Example Questions:
    - "What is reward hacking and why is it a problem?"
    - "How do language models hallucinate?"
    - "Explain diffusion models for video generation"
    - "Compare hallucination mitigation techniques"
    
    ### Agent Features:
    - **Smart Retrieval**: Decides when to search documents vs. answer directly
    - **Document Grading**: Evaluates relevance of retrieved information  
    - **Multi-attempt**: Will re-retrieve if initial results aren't good enough
    - **Source Citation**: References specific documents in answers
    """)