import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import json
import io
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
import tempfile
import os
import numpy as np

# Import core modules only
from src.dwg_parser import DWGParser
from src.ai_analyzer import AIAnalyzer
from src.visualization_new import PlanVisualizer
from src.export_utils import ExportManager

# Configure page
st.set_page_config(
    page_title="AI Architectural Space Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stability
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    .stButton > button {
        width: 100%;
    }
    .analysis-complete {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        background-color: #f0f8f0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with proper defaults
def init_session_state():
    defaults = {
        'zones': [],
        'analysis_results': {},
        'file_loaded': False,
        'analysis_complete': False,
        'current_file': None,
        'upload_key': 0
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

def reset_for_new_file():
    """Reset analysis state for new file while preserving upload state"""
    st.session_state.analysis_results = {}
    st.session_state.analysis_complete = False
    # Don't reset file_loaded or zones here - let the upload process handle that

def load_dwg_file(uploaded_file):
    """Load and parse DWG file with proper error handling"""
    if uploaded_file is None:
        return False

    try:
        with st.spinner(f"Loading {uploaded_file.name}..."):
            # Read file content
            file_bytes = uploaded_file.getvalue()
            file_size_mb = len(file_bytes) / (1024 * 1024)

            if file_size_mb > 50:
                st.error("File too large. Please use a file smaller than 50MB.")
                return False

            # Parse with DWGParser
            parser = DWGParser()
            zones = parser.parse_file(file_bytes, uploaded_file.name)

            if zones and len(zones) > 0:
                # Update session state
                st.session_state.zones = zones
                st.session_state.file_loaded = True
                st.session_state.current_file = uploaded_file.name
                st.session_state.analysis_results = {}  # Clear old analysis
                st.session_state.analysis_complete = False

                st.success(f"✅ Successfully loaded {len(zones)} zones from '{uploaded_file.name}'")
                return True
            else:
                st.error("❌ No zones found in the file. Please ensure the file contains closed room boundaries.")
                return False

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("💡 Try converting to DXF format or use a different file.")
        return False

def run_analysis():
    """Run analysis with proper state management"""
    if not st.session_state.zones:
        st.error("No zones loaded. Please upload a file first.")
        return

    try:
        with st.spinner("🤖 Running AI analysis..."):
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize analyzer
            analyzer = AIAnalyzer(confidence_threshold=0.7)

            # Step 1: Room analysis
            status_text.text("Analyzing room types...")
            progress_bar.progress(33)

            room_analysis = analyzer.analyze_room_types(st.session_state.zones)

            # Step 2: Placement analysis
            status_text.text("Calculating optimal placements...")
            progress_bar.progress(66)

            params = {
                'box_size': (2.0, 1.5),
                'margin': 0.5,
                'allow_rotation': True,
                'smart_spacing': True
            }

            placement_analysis = analyzer.analyze_furniture_placement(
                st.session_state.zones, params
            )

            # Step 3: Compile results
            status_text.text("Finalizing results...")
            progress_bar.progress(100)

            # Create comprehensive results
            total_boxes = sum(len(spots) for spots in placement_analysis.values())

            results = {
                'rooms': room_analysis,
                'placements': placement_analysis,
                'parameters': params,
                'total_boxes': total_boxes,
                'analysis_type': 'complete',
                'timestamp': datetime.now().isoformat()
            }

            # Save to session state
            st.session_state.analysis_results = results
            st.session_state.analysis_complete = True

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            st.success(f"🎉 Analysis complete! Found {total_boxes} optimal placements across {len(room_analysis)} rooms.")

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.session_state.analysis_complete = False

def main():
    """Main application with stable flow"""

    # Header
    st.title("🏗️ AI Architectural Space Analyzer")
    st.markdown("**Professional DWG/DXF Analysis with AI-Powered Room Detection & Furniture Placement**")

    # Main layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📂 File Upload")

        # File uploader with unique key to prevent conflicts
        uploaded_file = st.file_uploader(
            "Select DWG/DXF file:",
            type=['dwg', 'dxf'],
            help="Upload architectural drawings for analysis",
            key=f"file_upload_{st.session_state.upload_key}"
        )

        # Load button
        if uploaded_file:
            st.write(f"📄 **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.1f} MB)")

            if st.button("🔍 Load & Parse File", type="primary", key="load_btn"):
                success = load_dwg_file(uploaded_file)
                if success:
                    st.rerun()

        # Analysis section
        if st.session_state.file_loaded:
            st.divider()
            st.subheader("🤖 AI Analysis")

            st.success(f"File loaded: {len(st.session_state.zones)} zones")

            # Analysis parameters
            with st.expander("⚙️ Analysis Parameters"):
                box_length = st.slider("Box Length (m)", 1.0, 5.0, 2.0, 0.1)
                box_width = st.slider("Box Width (m)", 1.0, 4.0, 1.5, 0.1)
                margin = st.slider("Margin (m)", 0.0, 2.0, 0.5, 0.1)
                confidence = st.slider("Confidence Threshold", 0.5, 0.9, 0.7, 0.05)

            # Analysis button
            if not st.session_state.analysis_complete:
                if st.button("🚀 Run Analysis", type="primary", key="analyze_btn"):
                    run_analysis()
            else:
                st.success("✅ Analysis Complete")
                if st.button("🔄 Re-run Analysis", key="rerun_btn"):
                    reset_for_new_file()
                    run_analysis()

        # New file button
        if st.session_state.file_loaded or st.session_state.analysis_complete:
            st.divider()
            if st.button("📁 Load New File", key="new_file_btn"):
                # Reset everything for new file
                st.session_state.zones = []
                st.session_state.analysis_results = {}
                st.session_state.file_loaded = False
                st.session_state.analysis_complete = False
                st.session_state.current_file = None
                st.session_state.upload_key += 1  # Force new file uploader
                st.rerun()

    with col2:
        if not st.session_state.file_loaded:
            # Welcome screen
            st.info("👆 Upload a DWG or DXF file to get started")

            st.subheader("🌟 Features")
            st.markdown("""
            - **DWG/DXF Parsing**: Native support for AutoCAD files
            - **AI Room Detection**: Automatic room type classification
            - **Furniture Placement**: Optimal layout calculations
            - **Interactive Visualization**: 2D/3D plan views
            - **Export Options**: PDF reports, CSV data, JSON export
            """)

        elif not st.session_state.analysis_complete:
            # File loaded but no analysis
            st.info("👈 Click 'Run Analysis' to analyze the loaded zones")

            # Show basic file info
            st.subheader("📊 File Overview")
            st.write(f"**Current File:** {st.session_state.current_file}")
            st.write(f"**Zones Found:** {len(st.session_state.zones)}")

            # Basic visualization of zones
            if st.session_state.zones:
                try:
                    visualizer = PlanVisualizer()
                    fig = visualizer.create_basic_plot(st.session_state.zones)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.write("Zone preview not available")

        else:
            # Analysis complete - show results
            display_analysis_results()

def display_analysis_results():
    """Display comprehensive analysis results"""
    results = st.session_state.analysis_results

    # Results header
    st.markdown('<div class="analysis-complete">', unsafe_allow_html=True)
    st.subheader("🎯 Analysis Results")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Boxes", results.get('total_boxes', 0))
    with col2:
        st.metric("Rooms Analyzed", len(results.get('rooms', {})))
    with col3:
        total_area = sum(room.get('area', 0) for room in results.get('rooms', {}).values())
        st.metric("Total Area", f"{total_area:.1f} m²")
    with col4:
        avg_confidence = np.mean([room.get('confidence', 0) for room in results.get('rooms', {}).values()])
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏠 Room Details", "📈 Visualization", "📤 Export"])

    with tab1:
        display_overview_tab(results)

    with tab2:
        display_room_details_tab(results)

    with tab3:
        display_visualization_tab()

    with tab4:
        display_export_tab(results)

def display_overview_tab(results):
    """Display overview statistics"""
    st.subheader("Room Type Distribution")

    # Room type chart
    room_types = [room.get('type', 'Unknown') for room in results.get('rooms', {}).values()]
    room_counts = pd.Series(room_types).value_counts()

    if not room_counts.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig_pie = go.Figure(data=[go.Pie(labels=room_counts.index, values=room_counts.values)])
            fig_pie.update_layout(title="Room Types", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_bar = go.Figure(data=[go.Bar(x=room_counts.index, y=room_counts.values)])
            fig_bar.update_layout(title="Room Type Counts", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

def display_room_details_tab(results):
    """Display detailed room information"""
    st.subheader("Room Analysis Details")

    # Create detailed table
    room_data = []
    for zone_name, room_info in results.get('rooms', {}).items():
        placements = results.get('placements', {}).get(zone_name, [])

        room_data.append({
            'Zone': zone_name,
            'Room Type': room_info.get('type', 'Unknown'),
            'Confidence': f"{room_info.get('confidence', 0):.1%}",
            'Area (m²)': f"{room_info.get('area', 0):.1f}",
            'Dimensions': f"{room_info.get('dimensions', [0, 0])[0]:.1f} × {room_info.get('dimensions', [0, 0])[1]:.1f}",
            'Boxes Placed': len(placements),
            'Layer': room_info.get('layer', 'Unknown')
        })

    if room_data:
        df = pd.DataFrame(room_data)
        st.dataframe(df, use_container_width=True)

def display_visualization_tab():
    """Display interactive visualization"""
    st.subheader("Interactive Plan View")

    try:
        visualizer = PlanVisualizer()

        # Visualization options
        col1, col2 = st.columns([3, 1])

        with col2:
            show_zones = st.checkbox("Show Room Zones", value=True)
            show_boxes = st.checkbox("Show Furniture", value=True)
            show_labels = st.checkbox("Show Labels", value=True)
            color_by_type = st.checkbox("Color by Type", value=True)

        with col1:
            if st.session_state.analysis_results:
                fig = visualizer.create_interactive_plot(
                    st.session_state.zones,
                    st.session_state.analysis_results,
                    show_zones=show_zones,
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    color_by_type=color_by_type
                )
            else:
                fig = visualizer.create_basic_plot(st.session_state.zones)

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

def display_export_tab(results):
    """Display export options"""
    st.subheader("Export Analysis Results")

    try:
        export_manager = ExportManager()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**PDF Report**")
            if st.button("📄 Generate PDF", key="pdf_btn"):
                with st.spinner("Generating PDF..."):
                    pdf_data = export_manager.generate_pdf_report(
                        st.session_state.zones, results
                    )

                    st.download_button(
                        "📥 Download PDF",
                        data=pdf_data,
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )

        with col2:
            st.write("**JSON Data**")
            if st.button("📋 Generate JSON", key="json_btn"):
                json_data = export_manager.export_to_json(results)

                st.download_button(
                    "📥 Download JSON",
                    data=json_data,
                    file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col3:
            st.write("**CSV Statistics**")
            if st.button("📊 Generate CSV", key="csv_btn"):
                csv_data = export_manager.export_to_csv(results)

                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name=f"room_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        # Quick JSON preview
        with st.expander("👀 Preview Data Structure"):
            st.json({
                "total_rooms": len(results.get('rooms', {})),
                "total_placements": results.get('total_boxes', 0),
                "analysis_timestamp": results.get('timestamp', 'Unknown'),
                "sample_room": list(results.get('rooms', {}).values())[0] if results.get('rooms') else {}
            })

    except Exception as e:
        st.error(f"Export error: {str(e)}")

if __name__ == "__main__":
    main()