import streamlit as st
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DeepSeek Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configuration
with st.sidebar:
    st.header("Model Configuration")
    st.markdown("[Get HuggingFace Token](https://huggingface.co/settings/tokens)")

    # Dropdown to select model
    model_options = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ]
    selected_model = st.selectbox("Select Model", model_options, index=0)

    system_message = st.text_area(
        "System Message",
        value="You are a friendly chatbot created by ruslanmv.com. Provide clear, accurate, and brief answers. Keep responses polite, engaging, and to the point. If unsure, politely suggest alternatives.",
        height=100
    )

    max_tokens = st.slider(
        "Max Tokens",
        10, 4000, 100
    )

    temperature = st.slider(
        "Temperature",
        0.1, 4.0, 0.3
    )

    top_p = st.slider(
        "Top-p",
        0.1, 1.0, 0.6
    )

# Function to query the Hugging Face API
def query(payload, api_url):
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}
    logger.info(f"Sending request to {api_url} with payload: {payload}")
    response = requests.post(api_url, headers=headers, json=payload)
    logger.info(f"Received response: {response.status_code}, {response.text}")
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        logger.error(f"Failed to decode JSON response: {response.text}")
        return None

# Chat interface
st.title("🤖 DeepSeek Chatbot")
st.caption("Powered by Hugging Face Inference API - Configure in sidebar")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Generating response..."):
            # Prepare the payload for the API
            # Combine system message and user input into a single prompt
            full_prompt = f"{system_message}\n\nUser: {prompt}\nAssistant:"
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_full_text": False
                }
            }

            # Dynamically construct the API URL based on the selected model
            api_url = f"https://api-inference.huggingface.co/models/{selected_model}"
            logger.info(f"Selected model: {selected_model}, API URL: {api_url}")

            # Query the Hugging Face API using the selected model
            output = query(payload, api_url)

            # Handle API response
            if output is not None and isinstance(output, list) and len(output) > 0:
                if 'generated_text' in output[0]:
                    # Extract the assistant's response
                    assistant_response = output[0]['generated_text'].strip()

                    # Check for and remove duplicate responses
                    responses = assistant_response.split("\n</think>\n")
                    unique_response = responses[0].strip()

                    logger.info(f"Generated response: {unique_response}")

                    # Append response to chat only once
                    with st.chat_message("assistant"):
                        st.markdown(unique_response)

                    st.session_state.messages.append({"role": "assistant", "content": unique_response})
                else:
                    logger.error(f"Unexpected API response structure: {output}")
                    st.error("Error: Unexpected response from the model. Please try again.")
            else:
                logger.error(f"Empty or invalid API response: {output}")
                st.error("Error: Unable to generate a response. Please check the model and try again.")

    except Exception as e:
        logger.error(f"Application Error: {str(e)}", exc_info=True)
        st.error(f"Application Error: {str(e)}")