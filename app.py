import streamlit as st
import requests
import logging
import time
from typing import Dict, Any, Optional, List
import os
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from io import BytesIO
import hashlib
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import pickle
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize SBERT model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Vector store class
class SimpleVectorStore:
    def __init__(self, file_path: str = "vector_store.pkl"):
        self.file_path = file_path
        self.documents = []
        self.embeddings = []
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']

    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)

    def add_document(self, text: str, embedding: np.ndarray):
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        if not self.embeddings:
            return []
        
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

# Document processing functions
def process_text(text: str) -> List[str]:
    """Split text into chunks."""
    # Simple splitting by sentences (can be improved with better chunking)
    chunks = text.split('. ')
    return [chunk + '.' for chunk in chunks if chunk]

def process_image(image) -> str:
    """Extract text from image using OCR."""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return ""

def process_pdf(pdf_file) -> str:
    """Extract text from PDF."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file.flush()
            
            doc = fitz.open(tmp_file.name)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            os.unlink(tmp_file.name)
            return text
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return ""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_timestamps" not in st.session_state:
    st.session_state.request_timestamps = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = SimpleVectorStore()

# Rate limiting configuration
RATE_LIMIT_PERIOD = 60
MAX_REQUESTS_PER_PERIOD = 30

def check_rate_limit() -> bool:
    """Check if we're within rate limits."""
    current_time = time.time()
    st.session_state.request_timestamps = [
        ts for ts in st.session_state.request_timestamps 
        if current_time - ts < RATE_LIMIT_PERIOD
    ]
    
    if len(st.session_state.request_timestamps) >= MAX_REQUESTS_PER_PERIOD:
        return False
    
    st.session_state.request_timestamps.append(current_time)
    return True

def query(payload: Dict[str, Any], api_url: str) -> Optional[Dict[str, Any]]:
    """Query the Hugging Face API with error handling and rate limiting."""
    if not check_rate_limit():
        raise Exception(f"Rate limit exceeded. Please wait {RATE_LIMIT_PERIOD} seconds.")
    
    try:
        headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 429:
            raise Exception("Too many requests. Please try again later.")
        
        # response.raise_for_status()
        print(response)
        return response.json()
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"API request failed: {str(e)}")
        raise

def process_response(response: Dict[str, Any]) -> str:
    """Process and clean up the model's response."""
    if not isinstance(response, list) or not response:
        raise ValueError("Invalid response format")
    
    text = response[0]['generated_text'].strip()
    cleanup_patterns = [
        "Assistant:", "AI:", "</think>", "<think>",
        "\n\nHuman:", "\n\nUser:"
    ]
    for pattern in cleanup_patterns:
        text = text.replace(pattern, "").strip()
    
    return text

# Page configuration
st.set_page_config(
    page_title="RAG-Enabled DeepSeek Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Sidebar configuration
with st.sidebar:
    st.header("Model Configuration")
    st.markdown("[Get HuggingFace Token](https://huggingface.co/settings/tokens)")

    model_options = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ]
    selected_model = st.selectbox("Select Model", model_options, index=0)

    system_message = st.text_area(
        "System Message",
        value="You are a friendly chatbot with RAG capabilities. Use the provided context to answer questions accurately. If the context doesn't contain relevant information, say so.",
        height=100
    )

    max_tokens = st.slider("Max Tokens", 10, 4000, 100)
    temperature = st.slider("Temperature", 0.1, 4.0, 0.3)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.6)

    # File upload section
    st.header("Upload Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload files (PDF, Images, Text)", 
        type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
        accept_multiple_files=True
    )

# Process uploaded files
if uploaded_files:
    embedding_model = load_embedding_model()
    
    for file in uploaded_files:
        try:
            if file.type == "application/pdf":
                text = process_pdf(file)
            elif file.type.startswith("image/"):
                image = Image.open(file)
                text = process_image(image)
            else:  # text files
                text = file.getvalue().decode()

            chunks = process_text(text)
            for chunk in chunks:
                embedding = embedding_model.encode(chunk)
                st.session_state.vector_store.add_document(chunk, embedding)

            st.sidebar.success(f"Successfully processed {file.name}")
        except Exception as e:
            st.sidebar.error(f"Error processing {file.name}: {str(e)}")

# Main chat interface
st.title("🤖 RAG-Enabled DeepSeek Chatbot")
st.caption("Upload documents in the sidebar to enhance the chatbot's knowledge")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Type your message..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Generating response..."):
            # Get relevant context from vector store
            embedding_model = load_embedding_model()
            query_embedding = embedding_model.encode(prompt)
            relevant_contexts = st.session_state.vector_store.search(query_embedding)
            
            # Prepare context-enhanced prompt
            context_text = "\n".join(relevant_contexts)
            full_prompt = f"""Context information:
{context_text}

System: {system_message}

User: {prompt}
Assistant: Let me help you based on the provided context."""

            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_full_text": False
                }
            }

            api_url = f"https://api-inference.huggingface.co/models/{selected_model}"
            
            # Get and process response
            output = query(payload, api_url)
            if output:
                response_text = process_response(output)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                
                # Update chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text
                })

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        st.error(f"Error: {str(e)}")