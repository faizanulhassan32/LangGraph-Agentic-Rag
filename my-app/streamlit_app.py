import asyncio
import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph_sdk import get_client
from datasets import load_dataset
import threading
import nest_asyncio

# Enable nested event loops (important for Streamlit)
nest_asyncio.apply()

load_dotenv()

st.set_page_config(
    page_title="Multi-Hop RAG Chat",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def get_or_create_eventloop():
    """Get existing event loop or create new one if needed"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def run_async(coro):
    """Run async coroutine safely in Streamlit environment"""
    try:
        loop = get_or_create_eventloop()
        if loop.is_running():
            # If loop is already running, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except Exception as e:
        # Fallback: run in thread with new event loop
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()

@st.cache_resource
def get_langgraph_client():
    """Cached client creation to reuse connection"""
    try:
        return get_client(url="http://localhost:2024")
    except Exception as e:
        st.error(f"Could not connect to LangGraph server: {e}")
        return None

# --- Streamlit Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Use cached client
client = get_langgraph_client()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🛠️ Configuration")
    model = st.selectbox("Model", options=["gpt-4o-mini-2024-07-18", "gpt-4o"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 1000, 500, 50)
    
    st.markdown("---")
    
    st.header("📚 Dataset Management")
    st.info("The agent is designed for the HotpotQA dataset.")
    
    persist_dir = "./chroma_db_hotpotqa"
    if st.button("Download & Process HotpotQA"):
        with st.spinner("Downloading and processing HotpotQA... This may take a few minutes."):
            
            st.write("Loading dataset from Hugging Face...")
            dataset = load_dataset("hotpot_qa", "distractor", split="train", trust_remote_code=True).select(range(5000))
            st.write(f"Processing the first {len(dataset)} examples.")

            st.write("Extracting documents with full metadata...")
            all_docs = []
            seen_para_text = set()

            for example in dataset:
                top_level_metadata = {
                    "question_id": example.get('id', 'N/A'),
                    "question": example.get('question', 'N/A'),
                    "answer": example.get('answer', 'N/A'),
                    "question_type": example.get('type', 'N/A'),
                    "level": example.get('level', 'N/A')
                }

                titles = example['context']['title']
                sentences = example['context']['sentences']

                for i, para_sentences in enumerate(sentences):
                    para_text = "\n".join(para_sentences)

                    if para_text and para_text not in seen_para_text:
                        doc_metadata = top_level_metadata.copy()
                        doc_metadata["source_title"] = titles[i] if i < len(titles) else "Unknown Title"
                        
                        all_docs.append(Document(page_content=para_text, metadata=doc_metadata))
                        seen_para_text.add(para_text)
            
            st.write(f"Created {len(all_docs)} unique documents with rich metadata.")
            
            st.write("Chunking documents while preserving metadata...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(all_docs)
            st.write(f"Split documents into {len(chunks)} chunks.")

            st.write("Creating vector store...")
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
            
            st.success("HotpotQA dataset processed and stored with FULL metadata successfully!")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    if client:
        st.success("✅ Connected to LangGraph")
    else:
        st.error("❌ LangGraph server not connected")
        st.info("Run: `langgraph dev` to start the server")

# --- Main App UI ---
st.markdown('<h1 style="text-align: center;">🕵️‍♂️ Multi-Hop Agentic RAG</h1>', unsafe_allow_html=True)

if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.info("👋 Ask a complex question after processing the HotpotQA dataset!")

# --- Async Response Handler ---
async def get_agent_response_async(user_input, config, response_placeholder):
    """Async function to get agent response"""
    try:
        assistants = await client.assistants.search()
        assistant = None
        print(f"🔍 Streamlit: Found {len(assistants)} assistants total")
        
        for a in assistants:
            if a.get('name') == 'multi-hop-agent' or 'multi-hop' in str(a.get('graph_id', '')):
                assistant = a
                break
        
        if not assistant:
            return "Could not find the multi-hop-agent. Please ensure the LangGraph server is running."
        
        thread = await client.threads.create()
        print(f"✅ Streamlit: Thread created: {thread['thread_id']}")

        input_state = {"original_question": user_input}
        full_response = ""
        last_state = {}

        async for chunk in client.runs.stream(
            thread['thread_id'],
            assistant['assistant_id'],
            input=input_state,
            config=config,
            stream_mode="values"
        ):
            try:
                if hasattr(chunk, 'data') and chunk.data:
                    chunk_data = chunk.data
                    
                    if isinstance(chunk_data, dict):
                        last_state.update(chunk_data)
                        
                        if 'final_answer' in chunk_data and chunk_data['final_answer']:
                            full_response = chunk_data['final_answer']
                            print(f"🎉 Streamlit: Got final answer: {full_response[:100]}...")
                            response_placeholder.markdown(full_response)
                elif hasattr(chunk, 'event') and hasattr(chunk, 'data'):
                    if chunk.event == "values" and isinstance(chunk.data, dict):
                        last_state.update(chunk.data)
                        if 'final_answer' in chunk.data and chunk.data['final_answer']:
                            full_response = chunk.data['final_answer']
                            print(f"🎉 Streamlit: Got final answer from event: {full_response[:100]}...")
                            response_placeholder.markdown(full_response)
            except Exception as chunk_error:
                print(f"⚠️ Streamlit: Error processing chunk: {chunk_error}")
                continue
                
        if not full_response:
            return "I apologize, but I couldn't generate a complete response."
        
        return full_response

    except Exception as e:
        print(f"💥 Streamlit: Error occurred: {e}")
        return f"An error occurred: {e}"

def get_agent_response(user_input, config, response_placeholder):
    """Synchronous wrapper for async function"""
    return run_async(get_agent_response_async(user_input, config, response_placeholder))

# --- Chat Input Handler ---
if user_input := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not os.path.exists(persist_dir):
        st.warning("Please download and process the HotpotQA dataset first using the button in the sidebar.")
    elif not client:
        st.error("Cannot process request: LangGraph server is not connected.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🕵️‍♂️ The agent is on the case... This might take a moment."):
                config = {
                    "configurable": {
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                }
                response_placeholder = st.empty()
                print("🚀 Streamlit: Starting agent response...")
                
                # Use the synchronous wrapper instead of asyncio.run
                assistant_response = get_agent_response(user_input, config, response_placeholder)
                response_placeholder.markdown(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.rerun()

# --- Expander for Instructions ---
with st.expander("ℹ️ How to use & Sample Questions"):
    st.markdown("""
    ### Getting Started:
    1.  Make sure your `OPENAI_API_KEY` is set.
    2.  Start the LangGraph server in your terminal: `langgraph dev`
    3.  In the sidebar, click **"Download & Process HotpotQA Dataset"**.
    4.  Ask a question or use a sample below.

    ### 🎯 Sample HotpotQA Questions:
    -   `Which American film director, who directed the film that grossed the most money in 1975, was a co-founder of Amblin Entertainment?`
    -   `What is the name of the stadium that the team that David Beckham last played for, plays in?`
    -   `Arthur Ashe, who was the first winner of the ATP Player of the Year, has a stadium named after him at what U.S. Open complex?`
    """)