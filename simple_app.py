import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import tempfile
import os
from pathlib import Path

# Import only the real, working modules
from src.dwg_parser import DWGParser
from src.ai_analyzer import AIAnalyzer
from src.visualization import PlanVisualizer

# Configure page
st.set_page_config(
    page_title="DWG Analyzer - Real Version",
    page_icon="📐",
    layout="wide"
)

# Initialize session state
if 'zones' not in st.session_state:
    st.session_state.zones = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'dwg_loaded' not in st.session_state:
    st.session_state.dwg_loaded = False

def load_and_analyze_dwg(uploaded_file):
    """Load DWG file and perform real analysis"""
    try:
        with st.spinner("Loading and analyzing DWG file..."):
            # Validate file
            if not uploaded_file:
                st.error("No file selected")
                return
                
            file_ext = uploaded_file.name.lower().split('.')[-1]
            if file_ext not in ['dwg', 'dxf']:
                st.error(f"Unsupported format: {file_ext}")
                return
                
            # Save file temporarily
            with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Parse DWG file (REAL)
                parser = DWGParser()
                zones = parser.parse_file_from_path(tmp_file_path)
                
                if not zones:
                    st.warning("No zones found in file")
                    return
                
                # Analyze room types (REAL AI analysis)
                analyzer = AIAnalyzer()
                room_analysis = analyzer.analyze_room_types(zones)
                
                # Store results
                st.session_state.zones = zones
                st.session_state.analysis_results = room_analysis
                st.session_state.dwg_loaded = True
                
                st.success(f"Successfully analyzed {len(zones)} zones")
                st.rerun()
                
            finally:
                os.unlink(tmp_file_path)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    st.title("📐 DWG Analyzer - Real Analysis Tool")
    st.markdown("**This version contains only real, working functionality - no mock data**")
    
    # Sidebar
    with st.sidebar:
        st.header("Upload DWG/DXF File")
        
        uploaded_file = st.file_uploader(
            "Choose a DWG or DXF file",
            type=['dwg', 'dxf'],
            help="Maximum file size: 50MB"
        )
        
        if uploaded_file:
            if st.button("Analyze File"):
                load_and_analyze_dwg(uploaded_file)
        
        if st.session_state.dwg_loaded:
            st.success(f"File loaded: {len(st.session_state.zones)} zones")
    
    # Main content
    if not st.session_state.dwg_loaded:
        st.info("Upload a DWG or DXF file to begin analysis")
        
        # Show what's REAL vs MOCK
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ REAL Features")
            st.markdown("""
            - DWG/DXF file parsing
            - Zone extraction from drawings
            - Room type classification
            - Area calculations
            - Basic geometry analysis
            - Interactive visualization
            - Export to CSV/JSON
            """)
        
        with col2:
            st.subheader("❌ MOCK Features (Removed)")
            st.markdown("""
            - BIM model generation
            - Furniture catalog
            - Advanced AI optimization
            - CAD export
            - Multi-floor analysis
            - Collaboration features
            """)
    
    else:
        # Display real analysis results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Zone Visualization")
            
            # Create real visualization
            visualizer = PlanVisualizer()
            fig = visualizer.create_basic_plot(st.session_state.zones)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            if st.session_state.analysis_results:
                # Show real room analysis
                room_data = []
                for zone_id, room_info in st.session_state.analysis_results.items():
                    room_data.append({
                        'Zone': zone_id,
                        'Room Type': room_info.get('room_type', 'Unknown'),
                        'Confidence': f"{room_info.get('confidence', 0.0):.1%}",
                        'Area (m²)': f"{room_info.get('area', 0.0):.1f}",
                        'Width (m)': f"{room_info.get('width', 0.0):.1f}",
                        'Height (m)': f"{room_info.get('height', 0.0):.1f}"
                    })
                
                df = pd.DataFrame(room_data)
                st.dataframe(df, use_container_width=True)
                
                # Export options
                st.subheader("Export Data")
                
                if st.button("Download CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Analysis Results",
                        csv,
                        "dwg_analysis.csv",
                        "text/csv"
                    )

if __name__ == "__main__":
    main()