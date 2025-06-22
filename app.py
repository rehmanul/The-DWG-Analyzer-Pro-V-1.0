import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import tempfile
import os
from datetime import datetime

# Import custom modules
from src.dwg_parser import DWGParser
from src.ai_analyzer import AIAnalyzer
from src.visualization import PlanVisualizer
from src.export_utils import ExportManager
from src.optimization import PlacementOptimizer

# Import advanced features
from src.advanced_ai_models import AdvancedRoomClassifier, SemanticSpaceAnalyzer, OptimizationEngine
from src.bim_integration import BIMModelGenerator, BIMStandardsCompliance
from src.multi_floor_analysis import MultiFloorAnalyzer, FloorPlan
from src.collaborative_features import CollaborationManager, TeamPlanningInterface
from src.furniture_catalog import FurnitureCatalogManager
from src.cad_export import CADExporter

# Configure page
st.set_page_config(
    page_title="AI Architectural Space Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'zones' not in st.session_state:
    st.session_state.zones = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'placement_results' not in st.session_state:
    st.session_state.placement_results = {}
if 'dwg_loaded' not in st.session_state:
    st.session_state.dwg_loaded = False
if 'bim_model' not in st.session_state:
    st.session_state.bim_model = None
if 'furniture_configurations' not in st.session_state:
    st.session_state.furniture_configurations = []
if 'collaboration_active' not in st.session_state:
    st.session_state.collaboration_active = False
if 'multi_floor_project' not in st.session_state:
    st.session_state.multi_floor_project = None
if 'advanced_mode' not in st.session_state:
    st.session_state.advanced_mode = False

# Initialize advanced components
@st.cache_resource
def get_advanced_components():
    return {
        'advanced_classifier': AdvancedRoomClassifier(),
        'semantic_analyzer': SemanticSpaceAnalyzer(),
        'optimization_engine': OptimizationEngine(),
        'bim_generator': BIMModelGenerator(),
        'furniture_catalog': FurnitureCatalogManager(),
        'cad_exporter': CADExporter(),
        'collaboration_manager': CollaborationManager(),
        'multi_floor_analyzer': MultiFloorAnalyzer()
    }

def main():
    """Main application function with full advanced features"""
    
    # Get advanced components
    components = get_advanced_components()
    
    # Header with mode toggle
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🏗️ AI Architectural Space Analyzer PRO")
        st.markdown("**Complete Professional Solution for Architectural Analysis & Space Planning**")
    
    with col2:
        st.session_state.advanced_mode = st.toggle("Advanced Mode", value=st.session_state.advanced_mode)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📋 Control Panel")
        
        # Mode indicator
        mode_label = "🚀 Professional Mode" if st.session_state.advanced_mode else "🔧 Standard Mode"
        st.info(mode_label)
        
        # File upload section
        st.subheader("📂 Project Setup")
        
        if st.session_state.advanced_mode:
            project_type = st.selectbox("Project Type", [
                "Single Floor Analysis", 
                "Multi-Floor Building", 
                "BIM Integration Project",
                "Collaborative Team Project"
            ])
            
            if project_type == "Multi-Floor Building":
                setup_multi_floor_project()
            elif project_type == "Collaborative Team Project":
                setup_collaboration_project()
        
        uploaded_file = st.file_uploader(
            "Choose DWG/DXF files",
            type=['dwg', 'dxf'],
            accept_multiple_files=st.session_state.advanced_mode,
            help="Upload architectural plan files for analysis"
        )
        
        if uploaded_file is not None:
            if isinstance(uploaded_file, list):
                if st.button("🔍 Load Multiple Files", type="primary"):
                    load_multiple_dwg_files(uploaded_file)
            else:
                if st.button("🔍 Load & Parse File", type="primary"):
                    load_dwg_file(uploaded_file)
        
        st.divider()
        
        # Analysis parameters
        setup_analysis_parameters(components)
        
        st.divider()
        
        # Analysis controls
        if st.session_state.dwg_loaded:
            setup_analysis_controls(components)
    
    # Main content area
    if not st.session_state.dwg_loaded:
        display_welcome_screen()
    else:
        display_main_interface(components)

def load_dwg_file(uploaded_file):
    """Load and parse DWG/DXF file"""
    try:
        with st.spinner("🔄 Loading and parsing DWG file..."):
            # Save uploaded file temporarily
            file_bytes = uploaded_file.read()
            
            # Parse the DWG/DXF file
            parser = DWGParser()
            zones = parser.parse_file(file_bytes, uploaded_file.name)
            
            st.session_state.zones = zones
            st.session_state.dwg_loaded = True
            
            st.success(f"✅ Successfully loaded {len(zones)} zones from {uploaded_file.name}")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error loading DWG file: {str(e)}")

# Keep existing functions for backward compatibility
def run_ai_analysis(box_length, box_width, margin, confidence_threshold, enable_rotation, smart_spacing):
    """Run AI analysis on loaded zones"""
    try:
        with st.spinner("🤖 Running AI analysis..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize AI analyzer
            analyzer = AIAnalyzer(confidence_threshold)
            
            # Step 1: Room type analysis
            status_text.text("Analyzing room types...")
            progress_bar.progress(25)
            room_analysis = analyzer.analyze_room_types(st.session_state.zones)
            
            # Step 2: Furniture placement analysis
            status_text.text("Calculating optimal placements...")
            progress_bar.progress(50)
            
            params = {
                'box_size': (box_length, box_width),
                'margin': margin,
                'allow_rotation': enable_rotation,
                'smart_spacing': smart_spacing
            }
            
            placement_analysis = analyzer.analyze_furniture_placement(st.session_state.zones, params)
            
            # Step 3: Optimization
            status_text.text("Optimizing placements...")
            progress_bar.progress(75)
            
            optimizer = PlacementOptimizer()
            optimization_results = optimizer.optimize_placements(placement_analysis, params)
            
            # Step 4: Compile results
            status_text.text("Compiling results...")
            progress_bar.progress(100)
            
            st.session_state.analysis_results = {
                'rooms': room_analysis,
                'placements': placement_analysis,
                'optimization': optimization_results,
                'parameters': params,
                'total_boxes': sum(len(spots) for spots in placement_analysis.values())
            }
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ AI analysis complete! Found {st.session_state.analysis_results['total_boxes']} optimal box placements")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error during AI analysis: {str(e)}")

def display_analysis_results():
    """Display AI analysis results"""
    if not st.session_state.analysis_results:
        st.info("Run AI analysis to see results here")
        return
    
    results = st.session_state.analysis_results
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Boxes", results['total_boxes'])
    
    with col2:
        total_area = results['total_boxes'] * results['parameters']['box_size'][0] * results['parameters']['box_size'][1]
        st.metric("Total Area", f"{total_area:.1f} m²")
    
    with col3:
        efficiency = results['optimization'].get('total_efficiency', 0.85) * 100
        st.metric("Efficiency", f"{efficiency:.1f}%")
    
    with col4:
        num_rooms = len(results['rooms'])
        st.metric("Rooms Analyzed", num_rooms)
    
    st.divider()
    
    # Detailed room analysis
    st.subheader("🏠 Room Analysis")
    
    room_data = []
    for zone_name, room_info in results['rooms'].items():
        placements = results['placements'].get(zone_name, [])
        room_data.append({
            'Zone': zone_name,
            'Room Type': room_info['type'],
            'Confidence': f"{room_info['confidence']:.1%}",
            'Area (m²)': f"{room_info['area']:.1f}",
            'Dimensions': f"{room_info['dimensions'][0]:.1f} × {room_info['dimensions'][1]:.1f}",
            'Boxes Placed': len(placements),
            'Layer': room_info.get('layer', 'Unknown')
        })
    
    df = pd.DataFrame(room_data)
    st.dataframe(df, use_container_width=True)

def display_plan_visualization():
    """Display plan visualization"""
    if not st.session_state.zones:
        st.info("Load a DWG file to see visualization")
        return
    
    visualizer = PlanVisualizer()
    
    # Visualization options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎨 Display Options")
        show_zones = st.checkbox("Show Zones", value=True)
        show_boxes = st.checkbox("Show Box Placements", value=True)
        show_labels = st.checkbox("Show Labels", value=True)
        color_by_type = st.checkbox("Color by Room Type", value=True)
    
    with col1:
        # Generate visualization
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

def display_statistics():
    """Display detailed statistics"""
    if not st.session_state.analysis_results:
        st.info("Run AI analysis to see statistics")
        return
    
    results = st.session_state.analysis_results
    
    # Overall statistics
    st.subheader("📈 Overall Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Room type distribution
        room_types = [info['type'] for info in results['rooms'].values()]
        room_type_counts = pd.Series(room_types).value_counts()
        
        fig_pie = px.pie(
            values=room_type_counts.values,
            names=room_type_counts.index,
            title="Room Type Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Box placement by room
        placement_counts = {zone: len(placements) for zone, placements in results['placements'].items()}
        
        fig_bar = px.bar(
            x=list(placement_counts.keys()),
            y=list(placement_counts.values()),
            title="Boxes per Zone",
            labels={'x': 'Zone', 'y': 'Number of Boxes'}
        )
        fig_bar.update_xaxis(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)

def display_advanced_options():
    """Display advanced options and settings"""
    st.subheader("🔧 Advanced Settings")
    
    # Layer management
    if st.session_state.zones:
        st.subheader("📋 Layer Management")
        
        # Get all layers
        layers = set()
        for zone in st.session_state.zones:
            layers.add(zone.get('layer', 'Unknown'))
        
        # Layer selection
        selected_layers = st.multiselect(
            "Select layers to analyze",
            options=list(layers),
            default=list(layers)
        )
        
        if st.button("Update Layer Selection"):
            # Filter zones by selected layers
            filtered_zones = [zone for zone in st.session_state.zones if zone.get('layer', 'Unknown') in selected_layers]
            st.session_state.zones = filtered_zones
            st.success(f"Updated to {len(filtered_zones)} zones from selected layers")
            st.rerun()
    
    st.divider()
    
    # Export options
    st.subheader("📤 Export Options")
    
    if st.session_state.analysis_results:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Statistics (CSV)"):
                export_statistics_csv()
        
        with col2:
            if st.button("📋 Export Analysis (JSON)"):
                export_analysis_json()
        
        with col3:
            if st.button("📄 Generate PDF Report"):
                generate_pdf_report()

def export_statistics_csv():
    """Export statistics as CSV"""
    if not st.session_state.analysis_results:
        st.warning("No analysis results to export")
        return
    
    try:
        export_manager = ExportManager()
        csv_data = export_manager.export_to_csv(st.session_state.analysis_results)
        
        st.download_button(
            label="Download CSV Statistics",
            data=csv_data,
            file_name=f"analysis_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error exporting CSV: {str(e)}")

def export_analysis_json():
    """Export full analysis as JSON"""
    if not st.session_state.analysis_results:
        st.warning("No analysis results to export")
        return
    
    try:
        export_manager = ExportManager()
        json_data = export_manager.export_to_json(st.session_state.analysis_results)
        
        st.download_button(
            label="Download JSON Analysis",
            data=json_data,
            file_name=f"full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error exporting JSON: {str(e)}")

def generate_pdf_report():
    """Generate comprehensive PDF report"""
    if not st.session_state.analysis_results:
        st.warning("No analysis results to export")
        return
    
    try:
        with st.spinner("Generating PDF report..."):
            export_manager = ExportManager()
            pdf_data = export_manager.generate_pdf_report(st.session_state.zones, st.session_state.analysis_results)
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name=f"architectural_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")

def generate_report():
    """Generate quick report summary"""
    if not st.session_state.analysis_results:
        st.warning("No analysis results available")
        return
    
    results = st.session_state.analysis_results
    
    st.subheader("📋 Analysis Summary Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Project Overview:**")
        st.write(f"- Total zones analyzed: {len(st.session_state.zones)}")
        st.write(f"- Total box placements: {results['total_boxes']}")
        st.write(f"- Room types identified: {len(set(info['type'] for info in results['rooms'].values()))}")
        
    with col2:
        st.write("**Optimization Results:**")
        efficiency = results['optimization'].get('total_efficiency', 0.85) * 100
        st.write(f"- Layout efficiency: {efficiency:.1f}%")
        st.write(f"- Algorithm used: {results['optimization'].get('algorithm_used', 'Standard')}")
        
        # Calculate space utilization
        total_area = sum(info['area'] for info in results['rooms'].values())
        box_area = results['total_boxes'] * results['parameters']['box_size'][0] * results['parameters']['box_size'][1]
        utilization = (box_area / total_area * 100) if total_area > 0 else 0
        st.write(f"- Space utilization: {utilization:.1f}%")

if __name__ == "__main__":
    main()
    
    results = st.session_state.analysis_results
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Boxes", results['total_boxes'])
    
    with col2:
        total_area = results['total_boxes'] * results['parameters']['box_size'][0] * results['parameters']['box_size'][1]
        st.metric("Total Area", f"{total_area:.1f} m²")
    
    with col3:
        efficiency = results['optimization'].get('total_efficiency', 0.85) * 100
        st.metric("Efficiency", f"{efficiency:.1f}%")
    
    with col4:
        num_rooms = len(results['rooms'])
        st.metric("Rooms Analyzed", num_rooms)
    
    st.divider()
    
    # Detailed room analysis
    st.subheader("🏠 Room Analysis")
    
    room_data = []
    for zone_name, room_info in results['rooms'].items():
        placements = results['placements'].get(zone_name, [])
        room_data.append({
            'Zone': zone_name,
            'Room Type': room_info['type'],
            'Confidence': f"{room_info['confidence']:.1%}",
            'Area (m²)': f"{room_info['area']:.1f}",
            'Dimensions': f"{room_info['dimensions'][0]:.1f} × {room_info['dimensions'][1]:.1f}",
            'Boxes Placed': len(placements),
            'Layer': room_info.get('layer', 'Unknown')
        })
    
    df = pd.DataFrame(room_data)
    st.dataframe(df, use_container_width=True)
    
    # Box placement details
    st.subheader("📦 Box Placement Details")
    
    placement_data = []
    for zone_name, placements in results['placements'].items():
        for i, placement in enumerate(placements):
            placement_data.append({
                'Zone': zone_name,
                'Box ID': f"{zone_name}_Box_{i+1}",
                'Position (x, y)': f"({placement['position'][0]:.1f}, {placement['position'][1]:.1f})",
                'Size': f"{placement['size'][0]:.1f} × {placement['size'][1]:.1f}",
                'Suitability': f"{placement['suitability_score']:.2f}",
                'Area (m²)': f"{placement['area']:.1f}"
            })
    
    if placement_data:
        df_placements = pd.DataFrame(placement_data)
        st.dataframe(df_placements, use_container_width=True)
    else:
        st.info("No box placements found")

def display_plan_visualization():
    """Display plan visualization"""
    if not st.session_state.zones:
        st.info("Load a DWG file to see visualization")
        return
    
    visualizer = PlanVisualizer()
    
    # Visualization options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎨 Display Options")
        show_zones = st.checkbox("Show Zones", value=True)
        show_boxes = st.checkbox("Show Box Placements", value=True)
        show_labels = st.checkbox("Show Labels", value=True)
        color_by_type = st.checkbox("Color by Room Type", value=True)
    
    with col1:
        # Generate visualization
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
    
    # 3D Visualization
    if st.session_state.analysis_results and st.checkbox("Show 3D View"):
        st.subheader("🎯 3D Visualization")
        fig_3d = visualizer.create_3d_plot(
            st.session_state.zones,
            st.session_state.analysis_results
        )
        st.plotly_chart(fig_3d, use_container_width=True)

def display_statistics():
    """Display detailed statistics"""
    if not st.session_state.analysis_results:
        st.info("Run AI analysis to see statistics")
        return
    
    results = st.session_state.analysis_results
    
    # Overall statistics
    st.subheader("📈 Overall Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Room type distribution
        room_types = [info['type'] for info in results['rooms'].values()]
        room_type_counts = pd.Series(room_types).value_counts()
        
        fig_pie = px.pie(
            values=room_type_counts.values,
            names=room_type_counts.index,
            title="Room Type Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Box placement by room
        placement_counts = {zone: len(placements) for zone, placements in results['placements'].items()}
        
        fig_bar = px.bar(
            x=list(placement_counts.keys()),
            y=list(placement_counts.values()),
            title="Boxes per Zone",
            labels={'x': 'Zone', 'y': 'Number of Boxes'}
        )
        fig_bar.update_xaxis(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Efficiency metrics
    st.subheader("⚡ Efficiency Metrics")
    
    # Calculate various efficiency metrics
    total_room_area = sum(info['area'] for info in results['rooms'].values())
    total_box_area = results['total_boxes'] * results['parameters']['box_size'][0] * results['parameters']['box_size'][1]
    space_utilization = (total_box_area / total_room_area) * 100 if total_room_area > 0 else 0
    
    avg_suitability = 0
    if results['placements']:
        all_scores = []
        for placements in results['placements'].values():
            all_scores.extend([p['suitability_score'] for p in placements])
        avg_suitability = np.mean(all_scores) if all_scores else 0
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Space Utilization", f"{space_utilization:.1f}%")
    
    with metrics_col2:
        st.metric("Avg. Suitability Score", f"{avg_suitability:.2f}")
    
    with metrics_col3:
        st.metric("Boxes per m²", f"{results['total_boxes']/total_room_area:.2f}" if total_room_area > 0 else "0.00")

def display_advanced_options():
    """Display advanced options and settings"""
    st.subheader("🔧 Advanced Settings")
    
    # Layer management
    if st.session_state.zones:
        st.subheader("📋 Layer Management")
        
        # Get all layers
        layers = set()
        for zone in st.session_state.zones:
            layers.add(zone.get('layer', 'Unknown'))
        
        # Layer selection
        selected_layers = st.multiselect(
            "Select layers to analyze",
            options=list(layers),
            default=list(layers)
        )
        
        if st.button("Update Layer Selection"):
            # Filter zones by selected layers
            filtered_zones = [zone for zone in st.session_state.zones if zone.get('layer', 'Unknown') in selected_layers]
            st.session_state.zones = filtered_zones
            st.success(f"Updated to {len(filtered_zones)} zones from selected layers")
            st.rerun()
    
    st.divider()
    
    # Export options
    st.subheader("📤 Export Options")
    
    if st.session_state.analysis_results:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Statistics (CSV)"):
                export_statistics_csv()
        
        with col2:
            if st.button("📋 Export Analysis (JSON)"):
                export_analysis_json()
        
        with col3:
            if st.button("📄 Generate PDF Report"):
                generate_pdf_report()
    
    st.divider()
    
    # Debug information
    with st.expander("🔍 Debug Information"):
        if st.session_state.zones:
            st.write("**Loaded Zones:**", len(st.session_state.zones))
            st.write("**Analysis Results:**", bool(st.session_state.analysis_results))
            
            if st.checkbox("Show raw zone data"):
                st.json(st.session_state.zones[:2])  # Show first 2 zones as example

def export_statistics_csv():
    """Export statistics as CSV"""
    try:
        results = st.session_state.analysis_results
        
        # Create CSV data
        room_data = []
        for zone_name, room_info in results['rooms'].items():
            placements = results['placements'].get(zone_name, [])
            room_data.append({
                'Zone': zone_name,
                'Room_Type': room_info['type'],
                'Confidence': room_info['confidence'],
                'Area_m2': room_info['area'],
                'Width_m': room_info['dimensions'][0],
                'Height_m': room_info['dimensions'][1],
                'Boxes_Placed': len(placements),
                'Layer': room_info.get('layer', 'Unknown')
            })
        
        df = pd.DataFrame(room_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="architectural_analysis.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error exporting CSV: {str(e)}")

def export_analysis_json():
    """Export full analysis as JSON"""
    try:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert the results
        import json
        results_copy = json.loads(json.dumps(st.session_state.analysis_results, default=convert_numpy))
        
        json_data = json.dumps(results_copy, indent=2)
        
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name="architectural_analysis.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error exporting JSON: {str(e)}")

def generate_pdf_report():
    """Generate comprehensive PDF report"""
    try:
        export_manager = ExportManager()
        pdf_bytes = export_manager.generate_pdf_report(
            st.session_state.zones,
            st.session_state.analysis_results
        )
        
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name="architectural_analysis_report.pdf",
            mime="application/pdf"
        )
        
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")

def generate_report():
    """Generate quick report summary"""
    if not st.session_state.analysis_results:
        st.error("No analysis results to report")
        return
    
    results = st.session_state.analysis_results
    
    st.success("📊 Analysis Report Generated!")
    
    # Quick summary
    with st.container():
        st.markdown("### 📋 Summary Report")
        st.markdown(f"""
        **Analysis Complete**: {results['total_boxes']} optimal box placements found
        
        **Room Analysis**: {len(results['rooms'])} rooms analyzed
        - Average confidence: {np.mean([r['confidence'] for r in results['rooms'].values()]):.1%}
        
        **Box Parameters**: {results['parameters']['box_size'][0]}m × {results['parameters']['box_size'][1]}m
        
        **Total Coverage**: {results['total_boxes'] * results['parameters']['box_size'][0] * results['parameters']['box_size'][1]:.1f} m²
        
        **Algorithm**: {results['optimization']['algorithm_used']}
        """)

if __name__ == "__main__":
    main()
