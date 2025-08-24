"""UI components and display functions for Streamlit interface.

This module contains all the UI-related functions for the CM AGI Evaluation Platform,
including API key input handling, response display formatting, benchmark summary
visualization, and file download functionality.
"""

import streamlit as st
import os
import logging
from typing import Dict, Optional
from file_manager import get_latest_responses_file, get_latest_responses_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ui_components.log"),
        logging.StreamHandler()
    ]
)


def load_api_key() -> str:
    """Load OpenAI API key from environment variable or secure user input.
    
    First attempts to load the API key from the OPENAI_API_KEY environment variable.
    If not found, displays a password input field in the sidebar for the user to enter their key.
    
    Returns:
        str: The OpenAI API key, or an empty string if not provided
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    return api_key


def display_responses(responses_data: Dict):
    """Display API responses in a neat and organized manner using Streamlit components.
    
    Creates a summary section with metrics and then displays each prompt-response pair
    in expandable sections with metadata about tokens used and timestamp.
    
    Args:
        responses_data: Dictionary containing the responses to display
    """
    try:
        if not responses_data:
            st.warning("No responses found. Please generate responses first.")
            return
        
        # Validate required keys in responses_data
        required_keys = ["total_prompts", "timestamp", "responses"]
        for key in required_keys:
            if key not in responses_data:
                logging.error(f"Missing required key '{key}' in responses_data")
                st.error(f"Error: Response data is missing required information. Missing key: '{key}'")
                return
        
        st.header("ChatGPT API Responses")
        
        # Summary metrics
        try:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Prompts", responses_data["total_prompts"])
            with col2:
                total_tokens = sum(r.get("tokens_used", 0) for r in responses_data["responses"])
                st.metric("Total Tokens Used", total_tokens)
            with col3:
                timestamp = responses_data["timestamp"][:10] if len(responses_data["timestamp"]) >= 10 else responses_data["timestamp"]
                st.metric("Generated On", timestamp)
        except Exception as e:
            logging.error(f"Error displaying summary metrics: {str(e)}")
            st.warning("Could not display summary metrics due to an error in the data format.")
        
        st.divider()
        
        # Display each response
        for i, response in enumerate(responses_data["responses"]):
            try:
                # Validate required keys in response
                if "prompt" not in response or "prompt_number" not in response:
                    logging.warning(f"Response {i+1} missing required keys")
                    continue
                    
                prompt_preview = response['prompt'][:60] + "..." if len(response['prompt']) > 60 else response['prompt']
                with st.expander(f"🔸 Prompt {response['prompt_number']}: {prompt_preview}"):
                    st.markdown("**Prompt:**")
                    st.write(response["prompt"])
                    
                    st.markdown("**Response:**")
                    st.write(response.get("response", "No response data available"))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Tokens used: {response.get('tokens_used', 'N/A')}")
                    with col2:
                        timestamp = response.get('timestamp', '')
                        display_time = timestamp[:19] if timestamp and len(timestamp) >= 19 else timestamp
                        st.caption(f"Generated: {display_time}")
            except Exception as e:
                logging.error(f"Error displaying response {i+1}: {str(e)}")
                st.error(f"Could not display response {i+1} due to an error.")
    except Exception as e:
        logging.error(f"Error in display_responses: {str(e)}")
        st.error(f"An error occurred while displaying responses: {str(e)}")


def display_benchmark_summary(benchmark_data: Dict):
    """Display benchmark summary metrics in a row of Streamlit metric components.
    
    Shows key performance indicators including total duration, token usage,
    success rate, and tokens processed per second.
    
    Args:
        benchmark_data: Dictionary containing the benchmark metrics to display
    """
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Duration", f"{benchmark_data['total_duration_seconds']:.2f}s")
    with col2:
        st.metric("Total Tokens", benchmark_data['total_tokens_used'])
    with col3:
        st.metric("Success Rate", f"{benchmark_data['success_rate']:.1%}")
    with col4:
        st.metric("Tokens/Second", f"{benchmark_data['tokens_per_second']:.1f}")


def render_download_button():
    """Render download button for the latest responses file"""
    try:
        latest_file = get_latest_responses_file()
        latest_filename = get_latest_responses_filename()
        
        if latest_file and latest_filename:
            try:
                with open(latest_file, 'r') as f:
                    st.download_button(
                        label="Download Latest Responses JSON",
                        data=f.read(),
                        file_name=latest_filename,
                        mime="application/json"
                    )
                logging.info(f"Download button rendered for {latest_filename}")
            except IOError as e:
                logging.error(f"Error reading file for download: {str(e)}")
                st.error("Could not prepare file for download. The file may be missing or inaccessible.")
        else:
            st.info("No response files available for download.")
    except Exception as e:
        logging.error(f"Error in render_download_button: {str(e)}")
        st.error(f"An error occurred while preparing the download button: {str(e)}")
