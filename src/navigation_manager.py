"""
Navigation Manager for DWG Analyzer Application
Handles file management, session state, and workflow navigation
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional
import os


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
            'show_advanced_mode': False,
            'file_upload_key': 0,
            'processing_complete': False,
            'navigation_state': 'upload'  # upload, analysis, results
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
                st.warning("⏳ Analysis in progress")
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
            if st.button("🔄 New File", 
                       type="secondary", 
                       use_container_width=True):
                self.start_new_analysis()
                st.rerun()
        
        return None
    
    def display_sidebar_navigation(self):
        """Display minimal sidebar navigation menu"""
        # Keep sidebar minimal - no detailed analysis results
        return None
    
    def get_available_files(self) -> Dict[str, str]:
        """Get list of available DWG/DXF files in project directories"""
        sample_files = {}
        search_paths = [
            Path("attached_assets"),
            Path("."),
            Path("sample_files"),
            Path("uploads")
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for pattern in ["*.dwg", "*.dxf"]:
                    for file_path in search_path.glob(pattern):
                        if file_path.stat().st_size > 0:
                            display_name = file_path.stem.replace("_", " ").replace("-", " ").title()
                            sample_files[display_name] = str(file_path)
        
        return sample_files
    
    def display_file_browser(self):
        """Display enhanced file browser with preview"""
        st.subheader("📋 Available Files")
        
        available_files = self.get_available_files()
        
        if available_files:
            selected_file = st.selectbox(
                "Select from available files:",
                options=list(available_files.keys()),
                key=f"file_browser_{st.session_state.file_upload_key}",
                help="Choose from DWG/DXF files found in project directories"
            )
            
            if selected_file:
                file_path = available_files[selected_file]
                file_size = Path(file_path).stat().st_size
                st.caption(f"File size: {file_size / 1024:.1f} KB")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📂 Load Selected File", 
                               type="primary",
                               use_container_width=True):
                        return file_path
                
                with col2:
                    if st.button("🗑️ Remove from List", 
                               use_container_width=True):
                        # Could implement file removal logic here
                        st.info("File removal not implemented")
        
        else:
            st.info("No DWG/DXF files found in project directories")
            st.caption("Upload files or place them in the 'attached_assets' folder")
        
        return None
    
    def update_navigation_state(self, new_state: str):
        """Update navigation state"""
        valid_states = ['upload', 'analysis', 'results', 'export']
        if new_state in valid_states:
            st.session_state.navigation_state = new_state
    
    def get_navigation_state(self) -> str:
        """Get current navigation state"""
        return st.session_state.navigation_state
    
    def display_breadcrumb(self):
        """Display breadcrumb navigation"""
        steps = {
            'upload': '📁 Upload',
            'analysis': '🔍 Analysis', 
            'results': '📊 Results',
            'export': '📐 Export'
        }
        
        current_state = self.get_navigation_state()
        breadcrumb_items = []
        
        for step_key, step_label in steps.items():
            if step_key == current_state:
                breadcrumb_items.append(f"**{step_label}**")
            else:
                breadcrumb_items.append(step_label)
        
        st.caption(" → ".join(breadcrumb_items))