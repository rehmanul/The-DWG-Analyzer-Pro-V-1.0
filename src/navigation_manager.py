import streamlit as st
from typing import Optional, Dict, Any

class NavigationManager:
    """Manages application navigation and file workflow"""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize all required session state variables"""
        session_defaults = {
            'zones': [],
            'analysis_results': {},
            'furniture_configurations': [],
            'current_file': None,
            'current_project_id': None,
            'analysis_complete': False,
            'file_loaded': False,
            'dwg_loaded': False,
            'show_advanced_mode': False,
            'file_upload_key': 0,
            'processing_complete': False,
            'navigation_state': 'upload',  # upload, analysis, results
            'advanced_mode': False
        }

        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def reset_analysis_state(self):
        """Reset analysis-related session state for new file upload"""
        st.session_state.zones = []
        st.session_state.analysis_results = {}
        st.session_state.furniture_configurations = []
        st.session_state.analysis_complete = False
        st.session_state.processing_complete = False
        st.session_state.file_loaded = False
        st.session_state.dwg_loaded = False
        st.session_state.current_file = None
        st.session_state.navigation_state = 'upload'

    def display_navigation_header(self):
        """Display navigation header with file status and controls"""
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            st.title("🏗️ AI Architectural Space Analyzer PRO")

        with col2:
            # File status indicator
            if st.session_state.current_file:
                st.success(f"📄 {st.session_state.current_file}")
            else:
                st.info("No file loaded")

        with col3:
            # Analysis status
            if st.session_state.analysis_complete:
                zones_count = len(st.session_state.zones)
                st.success(f"✅ {zones_count} zones analyzed")
            elif st.session_state.zones:
                st.warning("⏳ Ready for analysis")
            else:
                st.info("Ready for upload")

        with col4:
            # New analysis button
            if st.button("🔄 New Analysis", 
                        help="Start fresh with a new file",
                        type="secondary"):
                self.start_new_analysis()
                st.rerun()

    def start_new_analysis(self):
        """Start a new analysis workflow"""
        self.reset_analysis_state()
        # Increment file upload key to reset file uploader
        st.session_state.file_upload_key += 1
        st.success("Ready for new file upload")

    def display_workflow_progress(self):
        """Display current workflow progress"""
        if st.session_state.navigation_state == 'upload':
            progress_value = 0.1
            status_text = "Upload file to begin analysis"
        elif st.session_state.zones and not st.session_state.analysis_complete:
            progress_value = 0.5
            status_text = "File loaded, run analysis"
        elif st.session_state.analysis_complete:
            progress_value = 1.0
            status_text = "Analysis complete"
        else:
            progress_value = 0.0
            status_text = "Waiting for file upload"

        st.progress(progress_value)
        st.caption(status_text)

    def display_action_buttons(self):
        """Display contextual action buttons based on current state"""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.session_state.zones and not st.session_state.analysis_complete:
                if st.button("🔍 Run Analysis", 
                           type="primary", 
                           use_container_width=True):
                    return 'run_analysis'

        with col2:
            if st.session_state.analysis_complete:
                if st.button("📊 View Results", 
                           type="secondary", 
                           use_container_width=True):
                    return 'view_results'

        with col3:
            if st.session_state.analysis_complete:
                if st.button("📐 Export CAD", 
                           type="secondary", 
                           use_container_width=True):
                    return 'export_cad'

        with col4:
            if st.session_state.zones:
                if st.button("🔄 Reset", 
                           type="secondary", 
                           use_container_width=True):
                    self.start_new_analysis()
                    return 'reset'

        return None

    def display_sidebar_navigation(self):
        """Display sidebar navigation"""
        if st.session_state.zones:
            st.success(f"Zones: {len(st.session_state.zones)}")

        if st.session_state.analysis_results:
            total_items = st.session_state.analysis_results.get('total_boxes', 0)
            st.success(f"Items placed: {total_items}")

        return None

    def display_breadcrumb(self):
        """Display breadcrumb navigation"""
        breadcrumb_items = []

        if st.session_state.navigation_state == 'upload':
            breadcrumb_items = ["📁 Upload"]
        elif st.session_state.zones and not st.session_state.analysis_complete:
            breadcrumb_items = ["📁 Upload", "🔧 Configure"]
        elif st.session_state.analysis_complete:
            breadcrumb_items = ["📁 Upload", "🔧 Configure", "📊 Results"]

        if breadcrumb_items:
            st.markdown(" → ".join(breadcrumb_items))

    def get_navigation_state(self) -> str:
        """Get current navigation state"""
        return st.session_state.get('navigation_state', 'upload')

    def update_navigation_state(self, state: str):
        """Update navigation state"""
        st.session_state.navigation_state = state