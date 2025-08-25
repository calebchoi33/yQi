"""Tab management for yQi evaluation platform."""

import streamlit as st
from typing import Dict, List, Any
import json
from datetime import datetime

from core.config import AppConfig
from core.services import ServiceManager
from core.exceptions import APIError, ConfigurationError, ValidationError
from ui.components import UIComponents
from api_client import send_prompts_to_chatgpt
from file_manager import save_responses_to_json, save_benchmark_data, load_responses_from_json
from batch_manager import batch_manager
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_rating_interface import SimpleRatingInterface
from ui_components import display_responses, display_benchmark_summary, render_download_button


class TabManager:
    """Manages all application tabs."""
    
    def __init__(self, config: AppConfig, service_manager: ServiceManager):
        self.config = config
        self.service_manager = service_manager
        self.ui = UIComponents(config)
    
    def render_generate_tab(self, api_key: str):
        """Render the Generate Responses tab."""
        st.header("📝 Generate New Responses")
        
        # Prompt configuration
        system_prompt, prompts = self.ui.render_prompt_configuration()
        self.ui.render_prompt_preview(prompts)
        
        # Processing mode selector
        processing_mode = self.ui.render_processing_mode_selector()
        
        # Generate button
        button_text = "🚀 Generate Responses" if processing_mode == "Real-time" else "📦 Submit Batch Job"
        if st.button(button_text, type="primary"):
            if processing_mode == "Real-time":
                self._handle_generation(api_key, system_prompt, prompts)
            else:
                self._handle_batch_submission(api_key, system_prompt, prompts)
        
        # Batch job monitoring (always show if there are any jobs)
        all_jobs = batch_manager.get_all_jobs()
        if processing_mode == "Batch" or all_jobs:
            st.divider()
            self._render_batch_monitoring(api_key)
    
    def _handle_generation(self, api_key: str, system_prompt: str, prompts: list):
        """Handle the response generation process."""
        # Validation
        if not api_key:
            self.ui.render_error_state("api_key", "")
            return
        
        if not prompts:
            self.ui.render_error_state("no_prompts", "")
            return
        
        # Generation process
        with self.ui.render_loading_state("Sending prompts to ChatGPT-4o..."):
            try:
                responses_data, benchmark_data = send_prompts_to_chatgpt(prompts, api_key, system_prompt)
                responses_file = save_responses_to_json(responses_data)
                benchmark_file = save_benchmark_data(benchmark_data)
                
                # Data is automatically saved to JSON files by the API client
                self.ui.render_generation_results(
                    responses_file, benchmark_file, benchmark_data
                )
                display_benchmark_summary(benchmark_data)
                
            except Exception as e:
                self.ui.render_error_state("generation", str(e))
    
    def render_view_tab(self):
        """Render the View Responses tab."""
        st.header("📋 View Saved Responses")
        
        responses_data = load_responses_from_json()
        
        if responses_data:
            display_responses(responses_data)
            st.divider()
            render_download_button()
        else:
            st.info("No saved responses found. Generate some responses first!")
    
    def render_rating_tab(self):
        """Render the Rate Responses tab."""
        rating_interface = SimpleRatingInterface(self.config)
        rating_interface.render_rating_interface()
    
    def render_statistics_tab(self):
        """Render the Statistics tab."""
        rating_interface = SimpleRatingInterface(self.config)
        rating_interface.render_statistics_view()
    
    def _handle_batch_submission(self, api_key: str, system_prompt: str, prompts: list):
        """Handle batch job submission."""
        # Validation
        if not api_key:
            self.ui.render_error_state("api_key", "")
            return
        
        if not prompts:
            self.ui.render_error_state("no_prompts", "")
            return
        
        # Batch submission process
        with self.ui.render_loading_state("Submitting batch job to OpenAI..."):
            try:
                batch_job_id = batch_manager.submit_batch_job(
                    api_key=api_key,
                    system_prompt=system_prompt,
                    prompts=prompts
                )
                
                st.success(f"Batch job submitted successfully! Job ID: {batch_job_id[:8]}...")
                st.info("Your batch will be processed within 24 hours. Check the monitoring section below for updates.")
                
            except Exception as e:
                self.ui.render_error_state("batch_submission", str(e))
    
    def _render_batch_monitoring(self, api_key: str):
        """Render batch job monitoring interface."""
        active_jobs = batch_manager.get_active_jobs()
        completed_jobs = [job for job in batch_manager.get_all_jobs() if job.get('status') == 'completed']
        
        # Active jobs monitoring
        action = self.ui.render_batch_job_monitor(active_jobs)
        
        if action:
            if action['action'] == 'check_status':
                with st.spinner("Checking job status..."):
                    try:
                        status_info = batch_manager.update_job_status(
                            action['job']['batch_job_id'], api_key
                        )
                        st.success(f"Status updated: {status_info['status']}")
                        
                        # Auto-download if completed
                        if status_info['status'] == 'completed':
                            with st.spinner("Auto-downloading completed results..."):
                                try:
                                    responses_data, benchmark_data = batch_manager.download_results(
                                        action['job']['batch_job_id'], api_key
                                    )
                                    st.success("Results automatically downloaded and saved!")
                                    display_benchmark_summary(benchmark_data)
                                except Exception as e:
                                    st.warning(f"Auto-download failed: {e}. Use manual download button.")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error checking status: {e}")
            
            elif action['action'] == 'download':
                with st.spinner("Downloading and processing results..."):
                    try:
                        responses_data, benchmark_data = batch_manager.download_results(
                            action['job']['batch_job_id'], api_key
                        )
                        st.success("Results downloaded and saved successfully!")
                        
                        # Show results summary
                        display_benchmark_summary(benchmark_data)
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error downloading results: {e}")
        
        # Completed jobs section
        if completed_jobs:
            st.subheader("Completed Jobs")
            for job in completed_jobs[:5]:  # Show last 5 completed jobs
                with st.expander(f"Job {job['batch_job_id'][:8]}... - Completed", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Run ID:** {job['run_id']}")
                        st.write(f"**Model:** {job['model']}")
                        st.write(f"**Prompts:** {len(job['prompts'])}")
                    with col2:
                        completed_at = job.get('completed_at', 'Unknown')
                        if isinstance(completed_at, (int, float)):
                            # Convert timestamp to readable format
                            completed_at = datetime.fromtimestamp(completed_at).strftime('%Y-%m-%d %H:%M:%S')
                        elif isinstance(completed_at, str) and len(completed_at) > 19:
                            completed_at = completed_at[:19]
                        st.write(f"**Completed:** {completed_at}")
                        if job.get('responses_file'):
                            st.write("✅ Results saved to files")
                        else:
                            if st.button(f"Download Results", key=f"download_completed_{job['batch_job_id']}"):
                                with st.spinner("Downloading results..."):
                                    try:
                                        responses_data, benchmark_data = batch_manager.download_results(
                                            job['batch_job_id'], api_key
                                        )
                                        st.success("Results downloaded!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
