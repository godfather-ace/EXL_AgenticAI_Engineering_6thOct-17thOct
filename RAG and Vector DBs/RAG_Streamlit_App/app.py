### Import Libraries
from llama_index.core import (
    Settings, 
    VectorStoreIndex, 
    SimpleDirectoryReader
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import streamlit as st
from dotenv import load_dotenv
import os

### Load API Keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

### Configure LLM and Embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.text_splitter = SentenceSplitter(chunk_size=512)
Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

### Load Documents
documents = SimpleDirectoryReader(input_dir="sample_data").load_data()

### Build Index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

### Streamlit Config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

### Custom Adaptive CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Open+Sans&display=swap');

    /* Default (light mode) */
    body {
        background: linear-gradient(135deg, #f8fafc, #e0eafc, #cfdef3);
        color: #222;
    }

    .block-container {
        background: rgba(255, 255, 255, 0.7);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
    }

    h1 {
        color: #333;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-size: 2.6em;
        font-weight: 600;
        margin-bottom: 0.4em;
    }

    input {
        border-radius: 10px !important;
        border: 1px solid #aaa !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000 !important;
    }

    button {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white !important;
        border: none;
        border-radius: 12px !important;
        font-size: 1.1em;
        font-weight: 600;
        padding: 0.6em 1.4em;
        transition: 0.3s;
    }

    button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #2575fc, #6a11cb);
    }

    .response-box {
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 12px;
        padding: 1.2em;
        margin-top: 1.2em;
        font-size: 1.05em;
        font-family: 'Open Sans', sans-serif;
        color: #222;
    }
    </style>
""", unsafe_allow_html=True)

### Chat UI
st.title("🤖 RAG Chatbot")
st.markdown("<p style='text-align:center; font-size:18px;'>Ask me anything based on your uploaded documents!</p>", unsafe_allow_html=True)

query = st.text_input("💬 Enter your query below:")

if st.button("🚀 Submit Query"):
    if not query.strip():
        st.error("⚠️ Please provide a query.")
    else:
        try:
            response = query_engine.query(query)
            st.markdown(f"<div class='response-box'>🧠 <b>Answer:</b><br>{response}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")