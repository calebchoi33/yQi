"""Service management for yQi evaluation platform."""

import logging
from typing import Tuple
import streamlit as st

from .config import AppConfig

logger = logging.getLogger(__name__)


class ServiceManager:
    """Simplified service management for file-based storage."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def show_status(self):
        """Show application status in Streamlit sidebar."""
        st.sidebar.success("File-based storage active")
        st.sidebar.info("All data stored locally in JSON files")
    
    def get_tab_configuration(self) -> Tuple[list, bool]:
        """Get tab configuration - now includes rating with file storage."""
        tabs = ["Generate Responses", "View Responses", "Rate Responses", "Statistics"]
        return tabs, True
