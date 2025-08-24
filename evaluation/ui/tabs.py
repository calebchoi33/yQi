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
        
        # Generate button
        if st.button("🚀 Generate Responses", type="primary"):
            self._handle_generation(api_key, system_prompt, prompts)
    
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
