import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any

# Configure model (updated for local execution)
DEFAULT_SYSTEM_PROMPT = """You are a friendly Assistant. Provide clear, accurate, and brief answers. 
Keep responses polite, engaging, and to the point. If unsure, politely suggest alternatives."""

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Directly specify model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Page configuration
st.set_page_config(
    page_title="DeepSeek-AI R1",
    page_icon="🤖",
    layout="centered"
)

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_loaded" not in st.session_state:
        st.session_state.update({
            "model_loaded": False,
            "model": None,
            "tokenizer": None
        })

def load_model():
    """Load model and tokenizer with quantization"""
    if not st.session_state.model_loaded:
        with st.spinner("Loading model (this may take a minute)..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            st.session_state.update({
                "model": model,
                "tokenizer": tokenizer,
                "model_loaded": True
            })

def configure_sidebar() -> Dict[str, Any]:
    """Create sidebar components"""
    with st.sidebar:
        st.header("Configuration")
        return {
            "system_message": st.text_area("System Message", value=DEFAULT_SYSTEM_PROMPT, height=100),
            "max_tokens": st.slider("Max Tokens", 10, 4000, 512),
            "temperature": st.slider("Temperature", 0.1, 1.0, 0.7),
            "top_p": st.slider("Top-p", 0.1, 1.0, 0.9)
        }

def format_prompt(system_message: str, user_input: str) -> str:
    """Format prompt according to model's required template"""
    return f"""<|begin_of_sentence|>System: {system_message}<|User|>{user_input}<|Assistant|>"""

def generate_response(prompt: str, settings: Dict[str, Any]) -> str:
    """Generate response using local model"""
    inputs = st.session_state.tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    outputs = st.session_state.model.generate(
        inputs.input_ids,
        max_new_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        do_sample=True,
        pad_token_id=st.session_state.tokenizer.eos_token_id
    )
    
    response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("\n</think>\n")[0].strip()
    return response.split("<|Assistant|>")[-1].strip()

def handle_chat_interaction(settings: Dict[str, Any]):
    """Manage chat interactions"""
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("Generating response..."):
                full_prompt = format_prompt(
                    settings["system_message"],
                    prompt
                )
                
                response = generate_response(full_prompt, settings)
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        except Exception as e:
            st.error(f"Generation error: {str(e)}")

def display_chat_history():
    """Display chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    initialize_session_state()
    load_model()  # Load model before anything else
    settings = configure_sidebar()
    
    st.title("🤖 DeepSeek Chat")
    st.caption(f"Running {MODEL_NAME} directly on {DEVICE.upper()}")
    
    display_chat_history()
    handle_chat_interaction(settings)

if __name__ == "__main__":
    main()
