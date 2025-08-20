"""CM AGI Evaluation Platform - Main Streamlit Application.

This is the main entry point for the CM AGI Evaluation Platform, a Streamlit-based
application for evaluating Traditional Chinese Medicine AI capabilities using OpenAI's
ChatGPT API. The application provides a user-friendly interface for sending prompts,
viewing responses, and analyzing performance metrics.

The application is structured with a modular design, with separate modules for:
- Configuration (config.py)
- API interactions (api_client.py)
- File management (file_manager.py)
- UI components (ui_components.py)
"""

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import modules
from config import DEFAULT_PROMPTS
from api_client import send_prompts_to_chatgpt
from file_manager import save_responses_to_json, save_benchmark_data, load_responses_from_json
from ui_components import load_api_key, display_responses, display_benchmark_summary, render_download_button

# Configure page
st.set_page_config(
    page_title="CM AGI Evaluation Platform",
    page_icon="🏥",
    layout="wide"
)


def main():
    """Main application function that sets up the Streamlit UI and handles user interactions.
    
    Creates a two-tab interface:
    1. Generate Responses: For configuring and sending prompts to the API
    2. View Responses: For viewing and downloading previously generated responses
    """
    st.title("🏥 CM AGI Evaluation Platform")
    st.markdown("Evaluate Traditional Chinese Medicine AI capabilities using ChatGPT-4o with comprehensive benchmarking.")
    
    # Sidebar for API key and controls
    st.sidebar.header("Configuration")
    api_key = load_api_key()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["Generate Responses", "View Responses"])
    
    with tab1:
        st.header("📝 Generate New Responses")
        
        # Prompt configuration
        st.subheader("Configure Prompts")
        use_default = st.checkbox("Use default prompts", value=True)
        
        if use_default:
            prompts = DEFAULT_PROMPTS
            st.info("Using 20 default Chinese medical case prompts for CM AGI evaluation.")
        else:
            st.write("Enter your custom prompts (one per line):")
            custom_prompts = st.text_area("Custom Prompts", height=200)
            prompts = [p.strip() for p in custom_prompts.split('\n') if p.strip()]
        
        # Display prompts to be sent
        if prompts:
            st.subheader("Prompts to Send:")
            for i, prompt in enumerate(prompts, 1):
                st.write(f"{i}. {prompt}")
        
        # Generate responses button
        if st.button("🚀 Generate Responses", type="primary"):
            if not api_key:
                st.error("Please provide your OpenAI API key.")
            elif not prompts:
                st.error("Please provide at least one prompt.")
            else:
                with st.spinner("Sending prompts to ChatGPT-4o..."):
                    responses_data, benchmark_data = send_prompts_to_chatgpt(prompts, api_key)
                    responses_file = save_responses_to_json(responses_data)
                    benchmark_file = save_benchmark_data(benchmark_data)
                    
                    st.success("✅ Responses generated and saved successfully!")
                    st.info(f"📄 Responses saved to: {responses_file}")
                    st.info(f"📊 Benchmark data saved to: {benchmark_file}")
                    
                    # Display benchmark summary
                    display_benchmark_summary(benchmark_data)
                    
                    st.balloons()
    
    with tab2:
        st.header("📋 View Saved Responses")
        
        # Load and display responses
        responses_data = load_responses_from_json()
        
        if responses_data:
            display_responses(responses_data)
            
            # Download button for JSON file
            st.divider()
            render_download_button()
        else:
            st.info("No saved responses found. Generate some responses first!")


if __name__ == "__main__":
    main()
