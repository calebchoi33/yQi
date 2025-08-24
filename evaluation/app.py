"""yQi Evaluation Platform - Main Streamlit Application"""

import streamlit as st
from core import AppConfig, ServiceManager
from ui import UIComponents, TabManager

# Configure page
st.set_page_config(
    page_title="CM AGI Evaluation Platform",
    page_icon="🏥",
    layout="wide"
)
 

def main():
    """Main application function with clean architecture.
    
    Features:
    - Centralized configuration management
    - Service factory pattern
    - Clean separation of concerns
    - Graceful error handling
    """
    # Initialize configuration and services
    config = AppConfig()
    service_manager = ServiceManager(config)
    ui = UIComponents(config)
    tab_manager = TabManager(config, service_manager)
    
    # Page setup
    st.title("🏥 yQi Evaluation Platform")
    st.markdown("Evaluate Traditional Chinese Medicine AI capabilities with async processing and file-based rating system.")
    
    # Sidebar configuration
    api_key = ui.render_sidebar_config()
    service_manager.show_status()
    
    # Dynamic tab configuration
    tab_names, has_database = service_manager.get_tab_configuration()
    tabs = st.tabs(tab_names)
    
    # Render tabs
    with tabs[0]:
        tab_manager.render_generate_tab(api_key)
    
    with tabs[1]:
        tab_manager.render_view_tab()
    
    # Rating and statistics tabs (now file-based)
    with tabs[2]:
        tab_manager.render_rating_tab()
    
    with tabs[3]:
        tab_manager.render_statistics_tab()


if __name__ == "__main__":
    main()
