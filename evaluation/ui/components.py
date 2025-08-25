"""Reusable UI components for yQi evaluation platform."""

import streamlit as st
import os
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from core.config import AppConfig
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


class UIComponents:
    """Collection of reusable UI components."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def render_sidebar_config(self) -> Optional[str]:
        """Render sidebar configuration and return API key."""
        st.sidebar.header("Configuration")
        
        # API Key input
        api_key = self.config.api.api_key
        if not api_key:
            api_key = st.sidebar.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key"
            )
        else:
            st.sidebar.success("API Key loaded from environment")
        
        st.sidebar.divider()
        
        # Model Selection
        st.sidebar.subheader("Model Settings")
        selected_model = st.sidebar.selectbox(
            "Select Model",
            ["Default"],
            index=0,
            help="Choose the AI model for evaluation (more options coming soon)"
        )
        
        # Store selected model in session state for future use
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "Default"
        st.session_state.selected_model = selected_model
        
        return api_key
    
    def render_prompt_configuration(self) -> tuple:
        """Render prompt configuration section with editable prompts."""
        st.subheader("Configure Prompts")
        
        # Import here to avoid circular imports
        from prompts import DEFAULT_PROMPTS
        
        # Initialize session state for prompts if not exists
        if 'editable_prompts' not in st.session_state:
            st.session_state.editable_prompts = DEFAULT_PROMPTS.copy()
        
        # Initialize system prompt if not exists
        if 'system_prompt' not in st.session_state:
            st.session_state.system_prompt = """You are an expert Traditional Chinese Medicine (TCM) practitioner with extensive knowledge of TCM diagnosis, treatment principles, and herbal prescriptions. 

For each medical case presented, please provide:
1. A detailed TCM diagnostic analysis including symptom analysis and syndrome differentiation (辨證論治)
2. Specific Chinese herbal prescriptions with individual herbs and recommended dosages

Please respond in a structured, professional manner that would be appropriate for clinical practice."""
        
        # System Prompt Configuration
        st.subheader("System Prompt")
        st.write("This prompt provides context to the AI for all subsequent medical case prompts:")
        
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            height=150,
            help="This prompt sets the context and instructions for the AI before processing medical cases",
            key="system_prompt_input"
        )
        st.session_state.system_prompt = system_prompt
        
        st.divider()
        
        # Medical Case Prompts Configuration
        st.subheader("Medical Case Prompts")
        
        # Configuration options
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            prompt_mode = st.radio(
                "Prompt Configuration",
                ["Edit Default Prompts", "Use Custom Prompts", "Upload JSON File"],
                horizontal=True
            )
        
        with col2:
            if st.button("Reset to Defaults"):
                st.session_state.editable_prompts = DEFAULT_PROMPTS.copy()
                st.rerun()
        
        with col3:
            if st.button("Add New Prompt"):
                st.session_state.editable_prompts.append("")
                st.rerun()
        
        with col4:
            if st.button("Export JSON"):
                self._export_prompts_json()
                st.rerun()
        
        if prompt_mode == "Edit Default Prompts":
            st.write("**Edit individual prompts below:**")
            
            # Create editable text areas for each prompt
            for i, prompt in enumerate(st.session_state.editable_prompts):
                col1, col2 = st.columns([10, 1])
                
                with col1:
                    new_prompt = st.text_area(
                        f"Prompt {i+1}",
                        value=prompt,
                        height=100,
                        key=f"prompt_{i}",
                        help=f"Edit prompt {i+1}"
                    )
                    st.session_state.editable_prompts[i] = new_prompt
                
                with col2:
                    st.write("")  # Spacing
                    st.write("")  # Spacing
                    if st.button("Delete", key=f"delete_{i}", help=f"Delete prompt {i+1}"):
                        if len(st.session_state.editable_prompts) > 1:
                            st.session_state.editable_prompts.pop(i)
                            st.rerun()
                        else:
                            st.error("Cannot delete the last prompt!")
            
            # Filter out empty prompts
            prompts = [p.strip() for p in st.session_state.editable_prompts if p.strip()]
            
        elif prompt_mode == "Use Custom Prompts":
            st.write("**Enter your custom prompts (one per line):**")
            custom_prompts = st.text_area(
                "Custom Prompts", 
                height=300,
                help="Enter each prompt on a separate line"
            )
            prompts = [p.strip() for p in custom_prompts.split('\n') if p.strip()]
        
        elif prompt_mode == "Upload JSON File":
            # JSON file upload mode
            st.write("**Upload a JSON file with system prompt and prompts:**")
            
            # Show expected format
            with st.expander("Expected JSON Format", expanded=False):
                st.code('''
{
  "system_prompt": "Your system prompt here...",
  "prompts": [
    "First medical case prompt...",
    "Second medical case prompt...",
    "Third medical case prompt..."
  ]
}
                ''', language="json")
            
            uploaded_file = st.file_uploader(
                "Choose JSON file",
                type="json",
                help="Upload a JSON file containing system_prompt and prompts array"
            )
            
            if uploaded_file is not None:
                try:
                    import json
                    content = json.load(uploaded_file)
                    
                    # Validate JSON structure
                    if "system_prompt" in content and "prompts" in content:
                        # Update session state with uploaded data
                        st.session_state.system_prompt = content["system_prompt"]
                        st.session_state.editable_prompts = content["prompts"]
                        
                        st.success(f"Successfully loaded {len(content['prompts'])} prompts from file!")
                        st.info("System prompt and prompts have been updated. You can switch to 'Edit Default Prompts' to modify them.")
                        
                        # Use the uploaded prompts
                        prompts = content["prompts"]
                        system_prompt = content["system_prompt"]
                        
                    else:
                        st.error("Invalid JSON format. Please ensure your file contains 'system_prompt' and 'prompts' fields.")
                        prompts = []
                        
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON file: {e}")
                    prompts = []
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    prompts = []
            else:
                prompts = []
        
        # Show prompt count
        if prompts:
            st.success(f"{len(prompts)} medical case prompts configured")
        else:
            st.warning("No medical case prompts configured")
        
        return system_prompt, prompts
    
    def render_prompt_preview(self, prompts: list):
        """Render prompt preview section."""
        if prompts:
            with st.expander(f"Preview Prompts ({len(prompts)} total)", expanded=False):
                for i, prompt in enumerate(prompts[:3], 1):  # Show first 3
                    st.write(f"**{i}.** {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                if len(prompts) > 3:
                    st.write(f"... and {len(prompts) - 3} more prompts")
    
    def render_generation_results(self, responses_file: str, benchmark_file: str, 
                                benchmark_data: Dict[str, Any], evaluation_id: Optional[str] = None):
        """Render generation results section."""
        st.success("Responses generated and saved successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Responses: {responses_file}")
            st.info(f"Benchmarks: {benchmark_file}")
        
        with col2:
            if evaluation_id:
                st.info(f"Database ID: {evaluation_id}")
            
            # Key metrics
            if benchmark_data:
                duration = benchmark_data.get('total_duration_seconds', 0)
                success_rate = benchmark_data.get('success_rate', 0)
                st.metric("Duration", f"{duration:.1f}s")
                st.metric("Success Rate", f"{success_rate:.1%}")
    
    def render_error_state(self, error_type: str, message: str):
        """Render error state with appropriate styling."""
        if error_type == "api_key":
            st.error("Please provide your OpenAI API key.")
        elif error_type == "no_prompts":
            st.error("Please provide at least one prompt.")
        elif error_type == "database":
            st.warning(f"Database error: {message}")
        else:
            st.error(f"Error: {message}")
    
    def _export_prompts_json(self):
        """Export current system prompt and prompts as JSON for download."""
        import json
        from datetime import datetime
        
        # Prepare export data
        export_data = {
            "system_prompt": st.session_state.get('system_prompt', ''),
            "prompts": st.session_state.get('editable_prompts', []),
            "exported_at": datetime.now().isoformat(),
            "total_prompts": len(st.session_state.get('editable_prompts', []))
        }
        
        # Convert to JSON string
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        # Create download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"yqi_prompts_{timestamp}.json"
        
        st.download_button(
            label="Download JSON File",
            data=json_str,
            file_name=filename,
            mime="application/json",
            help="Download current system prompt and prompts as JSON file"
        )
    
    def render_loading_state(self, message: str = "Processing..."):
        """Render loading state with spinner."""
        return st.spinner(message)
    
    def render_metrics_row(self, metrics: Dict[str, Any]):
        """Render a row of metrics."""
        cols = st.columns(len(metrics))
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(label, value)
    
    def render_processing_mode_selector(self):
        """Render processing mode selector (Real-time vs Batch)."""
        st.subheader("Processing Mode")
        
        mode = st.radio(
            "Choose processing mode:",
            options=["Real-time", "Batch"],
            help="Real-time: Get responses immediately (higher cost). Batch: Process asynchronously within 24h (50% cost savings)."
        )
        
        if mode == "Batch":
            st.info("Batch processing offers 50% cost savings and higher rate limits. Results will be available within 24 hours.")
        
        return mode
    
    def render_batch_job_monitor(self, batch_jobs: List[Dict]):
        """Render batch job monitoring interface."""
        if not batch_jobs:
            st.info("No active batch jobs.")
            return
        
        st.subheader("Active Batch Jobs")
        
        for job in batch_jobs:
            with st.expander(f"Job {job['batch_job_id'][:8]}... - {job['status'].title()}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Run ID:** {job['run_id']}")
                    st.write(f"**Model:** {job['model']}")
                    st.write(f"**Prompts:** {len(job['prompts'])}")
                    st.write(f"**Submitted:** {job['submitted_at'][:19]}")
                
                with col2:
                    st.write(f"**Status:** {job['status']}")
                    if job.get('completed_at'):
                        completed_at = job['completed_at']
                        if isinstance(completed_at, (int, float)):
                            # Convert timestamp to readable format
                            from datetime import datetime
                            completed_str = datetime.fromtimestamp(completed_at).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            # Already a string, just truncate
                            completed_str = str(completed_at)[:19]
                        st.write(f"**Completed:** {completed_str}")
                    
                    # Action buttons
                    if job['status'] == 'completed':
                        if st.button(f"Download Results", key=f"download_{job['batch_job_id']}"):
                            return {'action': 'download', 'job': job}
                    elif job['status'] in ['submitted', 'in_progress', 'validating']:
                        if st.button(f"Check Status", key=f"check_{job['batch_job_id']}"):
                            return {'action': 'check_status', 'job': job}
        
        return None

    def display_responses(self, responses_data: Dict):
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

    def display_benchmark_summary(self, benchmark_data: Dict):
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

    def render_download_button(self):
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


# Legacy function wrappers for backward compatibility
def load_api_key() -> str:
    """Legacy function - use UIComponents.render_sidebar_config() instead."""
    import warnings
    warnings.warn("load_api_key() is deprecated. Use UIComponents.render_sidebar_config()", DeprecationWarning)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    return api_key


def display_responses(responses_data: Dict):
    """Legacy function - use UIComponents.display_responses() instead."""
    import warnings
    warnings.warn("display_responses() is deprecated. Use UIComponents.display_responses()", DeprecationWarning)
    from core.config import AppConfig
    ui = UIComponents(AppConfig())
    ui.display_responses(responses_data)


def display_benchmark_summary(benchmark_data: Dict):
    """Legacy function - use UIComponents.display_benchmark_summary() instead."""
    import warnings
    warnings.warn("display_benchmark_summary() is deprecated. Use UIComponents.display_benchmark_summary()", DeprecationWarning)
    from core.config import AppConfig
    ui = UIComponents(AppConfig())
    ui.display_benchmark_summary(benchmark_data)


def render_download_button():
    """Legacy function - use UIComponents.render_download_button() instead."""
    import warnings
    warnings.warn("render_download_button() is deprecated. Use UIComponents.render_download_button()", DeprecationWarning)
    from core.config import AppConfig
    ui = UIComponents(AppConfig())
    ui.render_download_button()
