import streamlit as st
import requests
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYSTEM_PROMPT = """You are a friendly Assistant. Provide clear, accurate, and brief answers. 
Keep responses polite, engaging, and to the point. If unsure, politely suggest alternatives."""

MODEL_OPTIONS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1"
]
API_BASE_URL = "https://api-inference.huggingface.co/models/"

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
    if "api_failures" not in st.session_state:
        st.session_state.api_failures = 0

def configure_sidebar() -> Dict[str, Any]:
    """Create sidebar components and return settings"""
    with st.sidebar:
        st.header("Model Configuration")
        st.markdown("[Get HuggingFace Token](https://huggingface.co/settings/tokens)")

        return {
            "model": st.selectbox("Select Model", MODEL_OPTIONS, index=0),
            "system_message": st.text_area(
                "System Message",
                value=DEFAULT_SYSTEM_PROMPT,
                height=100
            ),
            "max_tokens": st.slider("Max Tokens", 10, 4000, 100),
            "temperature": st.slider("Temperature", 0.1, 4.0, 0.3),
            "top_p": st.slider("Top-p", 0.1, 1.0, 0.6),
            "debug_chat": st.toggle("Return Full Text (Debugging Only)")
        }

def format_deepseek_prompt(system_message: str, user_input: str) -> str:
    """Format the prompt according to DeepSeek's required structure"""
    return f"""System: {system_message}
<|User|>{user_input}<|Assistant|>"""

def query_hf_api(payload: Dict[str, Any], api_url: str) -> Optional[Dict[str, Any]]:
    """Handle API requests with improved error handling"""
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        st.error(f"API Error: {e.response.status_code} - {e.response.text[:200]}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        st.error("Connection error. Please check your internet connection.")
    return None

def handle_chat_interaction(settings: Dict[str, Any]):
    """Manage chat input/output and API communication"""
    if prompt := st.chat_input("Type your message..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.spinner("Generating response..."):
                # Format prompt according to model requirements
                full_prompt = format_deepseek_prompt(
                    system_message=settings["system_message"],
                    user_input=prompt
                )

                payload = {
                    "inputs": full_prompt,
                    "parameters": {
                        "max_new_tokens": settings["max_tokens"],
                        "temperature": settings["temperature"],
                        "top_p": settings["top_p"],
                        "return_full_text": settings["debug_chat"],
                    }
                }

                api_url = f"{API_BASE_URL}{settings['model']}"
                output = query_hf_api(payload, api_url)

                if output and isinstance(output, list):
                    if 'generated_text' in output[0]:
                        response_text = output[0]['generated_text'].strip()
                        # Remove any remaining special tokens
                        response_text = response_text.split("\n</think>\n")[0].strip()
                        
                        # Display and store response
                        with st.chat_message("assistant"):
                            st.markdown(response_text)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_text}
                        )
                        return
                    
            # Handle failed responses
            st.session_state.api_failures += 1
            if st.session_state.api_failures > 2:
                st.error("Persistent API failures. Please check your API token and model selection.")
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            st.error("An unexpected error occurred. Please try again.")

def display_chat_history():
    """Render chat message history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    """Main application flow"""
    initialize_session_state()
    settings = configure_sidebar()
    
    st.title("🤖 DeepSeek Chatbot")
    st.caption(f"Current Model: {settings['model']}")
    st.caption("Powered by Hugging Face Inference API - Configure in sidebar")
    
    display_chat_history()
    handle_chat_interaction(settings)

if __name__ == "__main__":
    main()
