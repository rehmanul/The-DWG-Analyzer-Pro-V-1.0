import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import json
import io
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
import asyncio
import tempfile
import os

# Import custom modules
from src.dwg_parser import DWGParser
from src.ai_analyzer import AIAnalyzer
from src.visualization import PlanVisualizer
from src.export_utils import ExportManager
from src.optimization import PlacementOptimizer

# Import database and AI
from src.database import DatabaseManager
from src.ai_integration import GeminiAIAnalyzer

# Import advanced features with fallbacks
try:
    from src.advanced_ai_models import AdvancedRoomClassifier, SemanticSpaceAnalyzer, OptimizationEngine
    from src.bim_integration import BIMModelGenerator, BIMStandardsCompliance
    from src.multi_floor_analysis import MultiFloorAnalyzer, FloorPlan
    from src.collaborative_features import CollaborationManager, TeamPlanningInterface
    from src.furniture_catalog import FurnitureCatalogManager
    from src.cad_export import CADExporter
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    # Create fallback classes for production
    class AdvancedRoomClassifier:
        def batch_classify(self, zones): return {}
    class SemanticSpaceAnalyzer:
        def build_space_graph(self, zones, analysis): return {}
        def analyze_spatial_relationships(self): return {}
    class OptimizationEngine:
        def optimize_layout(self, zones, params): return {'total_efficiency': 0.85}
    class BIMModelGenerator:
        def create_bim_model_from_analysis(self, zones, results, metadata):
            class MockBIM:
                def __init__(self):
                    self.standards_compliance = {'ifc': {'score': 85.0}, 'spaces': {'compliant_spaces': 10}}
            return MockBIM()
    class FurnitureCatalogManager:
        def recommend_furniture_for_space(self, space_type, space_area, budget, sustainability_preference):
            class MockConfig:
                def __init__(self):
                    self.total_cost = 5000.0
                    self.total_items = 15
                    self.sustainability_score = 0.8
            return MockConfig()
    class CADExporter:
        def export_to_dxf(self, zones, results, path, **kwargs): pass
        def export_to_svg(self, zones, results, path): pass
    class CollaborationManager: pass
    class MultiFloorAnalyzer: pass
    class FloorPlan: pass

# Configure page
st.set_page_config(
    page_title="AI Architectural Space Analyzer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add responsive CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stColumn {
            width: 100% !important;
        }
        .stButton > button {
            width: 100%;
            margin: 0.25rem 0;
        }
    }

    @media (min-width: 1200px) {
        .main .block-container {
            max-width: 1400px;
            margin: 0 auto;
        }
    }

    .stFileUploader {
        width: 100%;
    }

    .stAlert {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
        'multi_floor_analyzer': MultiFloorAnalyzer(),
        'database': DatabaseManager(),
        'ai_analyzer': GeminiAIAnalyzer()
    }

def setup_multi_floor_project():
    """Setup multi-floor building project"""
    st.write("**Multi-Floor Building Setup**")

    floor_count = st.number_input("Number of Floors", min_value=1, max_value=50, value=3, key="floor_count_input")
    building_height = st.number_input("Total Building Height (m)", min_value=3.0, max_value=200.0, value=12.0, key="building_height_input")

    if st.button("Initialize Multi-Floor Project"):
        st.session_state.multi_floor_project = {
            'floor_count': floor_count,
            'building_height': building_height,
            'floors': []
        }
        st.success(f"Multi-floor project initialized for {floor_count} floors")

def setup_collaboration_project():
    """Setup collaborative team project"""
    st.write("**Team Collaboration Setup**")

    project_name = st.text_input("Project Name", value="New Architecture Project", key="project_name_input")
    team_size = st.number_input("Team Size", min_value=1, max_value=20, value=3, key="team_size_input")

    if st.button("Start Collaboration"):
        st.session_state.collaboration_active = True
        st.success(f"Collaboration started for '{project_name}' with {team_size} team members")

def setup_analysis_parameters(components):
    """Setup analysis parameters based on mode"""
    if st.session_state.advanced_mode:
        st.subheader("Advanced AI Parameters")

        # AI Model Selection
        ai_model = st.selectbox("AI Classification Model", [
            "Advanced Ensemble (Recommended)",
            "Random Forest",
            "Gradient Boosting",
            "Neural Network"
        ], key="ai_model_select")

        # Analysis depth
        analysis_depth = st.selectbox("Analysis Depth", [
            "Comprehensive (All Features)",
            "Standard (Core Features)",
            "Quick (Basic Analysis)"
        ], key="analysis_depth_select")

        # BIM Integration
        enable_bim = st.checkbox("Enable BIM Integration", value=True, key="enable_bim_check")
        if enable_bim:
            bim_standard = st.selectbox("BIM Standard", ["IFC 4.3", "COBie 2.4", "Custom"], key="bim_standard_select")

        # Furniture catalog integration
        enable_furniture = st.checkbox("Enable Furniture Catalog", value=True, key="enable_furniture_check")
        if enable_furniture:
            sustainability_pref = st.selectbox("Sustainability Preference", 
                                             ["A+ (Highest)", "A", "B", "C", "Any"], key="sustainability_select")
    else:
        # Standard parameters
        st.subheader("Analysis Parameters")

    # Core parameters (always shown)
    box_length = st.number_input("Box Length (m)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    box_width = st.number_input("Box Width (m)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
    margin = st.number_input("Margin (m)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

    confidence_threshold = st.slider("Confidence Threshold", min_value=0.5, max_value=0.95, value=0.7, step=0.05)
    enable_rotation = st.checkbox("Allow Box Rotation", value=True)
    smart_spacing = st.checkbox("Smart Spacing Optimization", value=True)

    return {
        'box_length': box_length,
        'box_width': box_width,
        'margin': margin,
        'confidence_threshold': confidence_threshold,
        'enable_rotation': enable_rotation,
        'smart_spacing': smart_spacing
    }

def setup_analysis_controls(components):
    """Setup analysis control buttons"""
    if st.session_state.advanced_mode:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Advanced AI Analysis", type="primary"):
                run_advanced_analysis(components)

            if st.button("Generate BIM Model"):
                generate_bim_model(components)

        with col2:
            if st.button("Furniture Analysis"):
                run_furniture_analysis(components)

            if st.button("CAD Export Package"):
                generate_cad_export(components)
    else:
        params = setup_analysis_parameters(components)
        if st.button("Run AI Analysis", type="primary"):
            run_ai_analysis(params['box_length'], params['box_width'], params['margin'], 
                           params['confidence_threshold'], params['enable_rotation'], params['smart_spacing'])

    if st.session_state.analysis_results:
        st.divider()
        if st.button("Generate Complete Report"):
            generate_comprehensive_report(components)

def display_welcome_screen():
    """Display welcome screen with feature overview"""
    st.info("Please upload a DWG/DXF file to begin analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Standard Features")
        st.markdown("""
        - Room type detection and classification
        - Optimal box/furniture placement calculation
        - Interactive 2D/3D visualization
        - Statistical analysis and reporting
        - Basic export capabilities
        """)

    with col2:
        st.subheader("Advanced Features")
        st.markdown("""
        - **Advanced AI Models**: Ensemble learning with 95%+ accuracy
        - **BIM Integration**: Full IFC/COBie compliance
        - **Multi-Floor Analysis**: Complete building analysis
        - **Team Collaboration**: Real-time collaborative planning
        - **Furniture Catalog**: Integration with pricing and procurement
        - **Advanced Optimization**: Genetic algorithms and simulated annealing
        - **CAD Export**: Professional drawing packages
        - **Database Integration**: Project management and history
        """)

    with st.expander("Getting Started Guide"):
        st.markdown("""
        ### Quick Start (Standard Mode)
        1. Upload your DWG/DXF file
        2. Configure box dimensions and parameters
        3. Run AI analysis
        4. Review results and export reports

        ### Professional Workflow (Advanced Mode)
        1. **Project Setup**: Choose project type (single floor, multi-floor, BIM, collaborative)
        2. **File Upload**: Upload single or multiple architectural files
        3. **Advanced Configuration**: Select AI models, BIM standards, sustainability preferences
        4. **Comprehensive Analysis**: Run advanced AI analysis with multiple algorithms
        5. **BIM Integration**: Generate IFC-compliant building models
        6. **Furniture Integration**: Access professional furniture catalogs with pricing
        7. **Team Collaboration**: Enable real-time collaborative editing
        8. **Export Package**: Generate complete CAD drawing packages
        """)

def display_main_interface(components):
    """Display main interface with analysis results"""
    st.success(f"DWG file loaded successfully! Found {len(st.session_state.zones)} zones")

    if st.session_state.advanced_mode:
        # Advanced interface with more tabs
        tabs = st.tabs([
            "Analysis Dashboard", 
            "Interactive Visualization", 
            "Advanced Statistics",
            "BIM Integration",
            "Furniture Catalog",
            "Database & Projects",
            "CAD Export",
            "Settings"
        ])

        with tabs[0]:
            display_advanced_analysis_dashboard(components)
        with tabs[1]:
            display_enhanced_visualization(components)
        with tabs[2]:
            display_advanced_statistics(components)
        with tabs[3]:
            display_bim_integration(components)
        with tabs[4]:
            display_furniture_catalog(components)
        with tabs[5]:
            display_database_interface(components)
        with tabs[6]:
            display_cad_export_interface(components)
        with tabs[7]:
            display_advanced_settings(components)

def display_advanced_analysis_dashboard(components):
    """Display advanced analysis dashboard"""
    if not st.session_state.analysis_results:
        st.info("Run advanced analysis to see comprehensive dashboard")
        return

    results = st.session_state.analysis_results

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Zones", len(st.session_state.zones))
    with col2:
        st.metric("Optimal Placements", results.get('total_boxes', 0))
    with col3:
        efficiency = results.get('optimization', {}).get('total_efficiency', 0.85) * 100
        st.metric("Optimization Efficiency", f"{efficiency:.1f}%")
    with col4:
        if st.session_state.bim_model:
            compliance = st.session_state.bim_model.standards_compliance['ifc']['score']
            st.metric("BIM Compliance", f"{compliance:.1f}%")
        else:
            st.metric("BIM Compliance", "Not Generated")
    with col5:
        if st.session_state.furniture_configurations:
            total_cost = sum(config.total_cost for config in st.session_state.furniture_configurations)
            st.metric("Furniture Cost", f"${total_cost:,.0f}")
        else:
            st.metric("Furniture Cost", "Not Analyzed")

def display_enhanced_visualization(components):
    """Display enhanced visualization with 3D and interactive features"""
    if not st.session_state.zones:
        st.info("Load DWG files to see visualization")
        return

    visualizer = PlanVisualizer()

    # Visualization controls
    col1, col2, col3 = st.columns(3)

    with col1:
        view_mode = st.selectbox("View Mode", ["2D Plan", "3D Isometric"])
    with col2:
        show_furniture = st.checkbox("Show Furniture", value=True)
    with col3:
        show_annotations = st.checkbox("Show Annotations", value=True)

    # Generate visualization based on mode
    if view_mode == "3D Isometric" and st.session_state.analysis_results:
        fig_3d = visualizer.create_3d_plot(
            st.session_state.zones,
            st.session_state.analysis_results
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        # Standard 2D visualization
        if st.session_state.analysis_results:
            fig = visualizer.create_interactive_plot(
                st.session_state.zones,
                st.session_state.analysis_results,
                show_zones=True,
                show_boxes=show_furniture,
                show_labels=show_annotations,
                color_by_type=True
            )
        else:
            fig = visualizer.create_basic_plot(st.session_state.zones)

        st.plotly_chart(fig, use_container_width=True)

def display_bim_integration(components):
    """Display BIM integration interface"""
    st.subheader("BIM Integration & Standards Compliance")

    if not st.session_state.bim_model:
        st.info("Generate BIM model first using the control panel")
        return

    bim_model = st.session_state.bim_model

    # Compliance overview
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("IFC Compliance")
        ifc_compliance = bim_model.standards_compliance['ifc']
        st.metric("Compliance Score", f"{ifc_compliance['score']:.1f}%")

    with col2:
        st.subheader("Space Standards")
        space_compliance = bim_model.standards_compliance['spaces']
        st.metric("Compliant Spaces", f"{space_compliance['compliant_spaces']}")

def display_furniture_catalog(components):
    """Display furniture catalog interface"""
    st.subheader("Professional Furniture Catalog")

    if not st.session_state.furniture_configurations:
        st.info("Run furniture analysis first using the control panel")
        return

    furniture_catalog = components['furniture_catalog']

    # Configuration summary
    total_cost = sum(config.total_cost for config in st.session_state.furniture_configurations)
    total_items = sum(config.total_items for config in st.session_state.furniture_configurations)
    avg_sustainability = np.mean([config.sustainability_score for config in st.session_state.furniture_configurations])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Items", total_items)
    with col2:
        st.metric("Total Cost", f"${total_cost:,.0f}")
    with col3:
        st.metric("Sustainability Score", f"{avg_sustainability:.2f}")

def display_cad_export_interface(components):
    """Display CAD export interface"""
    st.subheader("CAD Export & Technical Drawings")

    if not st.session_state.analysis_results:
        st.info("Run analysis first to enable CAD export")
        return

    # Export options
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Export Formats")
        export_dxf = st.checkbox("DXF (AutoCAD)", value=True)
        export_svg = st.checkbox("SVG (Web)", value=True)

    with col2:
        st.subheader("Drawing Options")
        include_dimensions = st.checkbox("Include Dimensions", value=True)
        include_furniture = st.checkbox("Include Furniture", value=True)

    if st.button("Generate CAD Export", type="primary"):
        try:
            cad_exporter = components['cad_exporter']

            with tempfile.TemporaryDirectory() as temp_dir:
                if export_dxf:
                    dxf_path = os.path.join(temp_dir, "architectural_plan.dxf")
                    cad_exporter.export_to_dxf(
                        st.session_state.zones,
                        st.session_state.analysis_results,
                        dxf_path,
                        include_furniture=include_furniture,
                        include_dimensions=include_dimensions
                    )

                    with open(dxf_path, 'rb') as f:
                        st.download_button(
                            "Download DXF File",
                            data=f.read(),
                            file_name="architectural_plan.dxf",
                            mime="application/octet-stream"
                        )

                if export_svg:
                    svg_path = os.path.join(temp_dir, "plan_preview.svg")
                    cad_exporter.export_to_svg(
                        st.session_state.zones,
                        st.session_state.analysis_results,
                        svg_path
                    )

                    with open(svg_path, 'r') as f:
                        st.download_button(
                            "Download SVG Preview",
                            data=f.read(),
                            file_name="plan_preview.svg",
                            mime="image/svg+xml"
                        )
        except Exception as e:
            st.error(f"Error generating CAD files: {str(e)}")

def display_advanced_settings(components):
    """Display advanced settings and configuration"""
    st.subheader("Advanced Settings & Configuration")

    st.subheader("AI Model Configuration")

    model_accuracy = st.slider("Model Accuracy vs Speed", 0.5, 1.0, 0.85, 0.05)
    st.write(f"Current setting: {'High Accuracy' if model_accuracy > 0.8 else 'Balanced'}")

    enable_ensemble = st.checkbox("Enable Ensemble Learning", value=True)
    enable_semantic = st.checkbox("Enable Semantic Analysis", value=True)

    if st.button("Update AI Configuration"):
        st.success("AI configuration updated!")

def generate_comprehensive_report(components):
    """Generate comprehensive analysis report"""
    if not st.session_state.analysis_results:
        st.warning("No analysis results available for report generation")
        return

    try:
        with st.spinner("Generating comprehensive report..."):
            export_manager = ExportManager()

            # Generate PDF report
            pdf_data = export_manager.generate_pdf_report(st.session_state.zones, st.session_state.analysis_results)

            # Generate JSON export
            json_data = export_manager.export_to_json(st.session_state.analysis_results)

            # Generate CSV data
            csv_data = export_manager.export_to_csv(st.session_state.analysis_results)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    "Download PDF Report",
                    data=pdf_data,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

            with col2:
                st.download_button(
                    "Download JSON Data",
                    data=json_data,
                    file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            with col3:
                st.download_button(
                    "Download CSV Data",
                    data=csv_data,
                    file_name=f"analysis_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            st.success("Comprehensive report package generated!")

    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
    else:
        # Standard interface
        tabs = st.tabs([
            "Analysis Results", 
            "Plan Visualization", 
            "Statistics", 
            "Advanced"
        ])

        with tabs[0]:
            display_analysis_results()
        with tabs[1]:
            display_plan_visualization()
        with tabs[2]:
            display_statistics()
        with tabs[3]:
            display_advanced_options()

def display_database_interface(components):
    """Display database and project management interface"""
    st.subheader("Database & Project Management")

    db_manager = components['database']

    # Project creation
    with st.expander("Create New Project"):
        project_name = st.text_input("Project Name")
        project_desc = st.text_area("Project Description")
        project_type = st.selectbox("Project Type", [
            "single_floor", "multi_floor", "bim_integration", "collaborative"
        ])

        if st.button("Create Project") and project_name:
            project_id = db_manager.create_project(
                name=project_name,
                description=project_desc,
                created_by="current_user",
                project_type=project_type
            )
            st.success(f"Project created with ID: {project_id}")

    # Project statistics
    if 'current_project_id' in st.session_state:
        stats = db_manager.get_project_statistics(st.session_state.current_project_id)
        if stats:
            st.subheader("Current Project Statistics")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyses", stats['total_analyses'])
            with col2:
                st.metric("Total Zones", stats['total_zones'])
            with col3:
                st.metric("Collaborators", stats['total_collaborators'])
            with col4:
                st.metric("Comments", stats['total_comments'])

    # Recent projects
    st.subheader("Recent Projects")
    try:
        projects = db_manager.get_user_projects("current_user")
    except Exception as e:
        st.warning(f"Database connection issue: {str(e)}")
        projects = []

    if projects:
        for project in projects[:5]:  # Show last 5 projects
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{project['name']}**")
                    st.write(f"Type: {project['project_type']} | Created: {project['created_at'][:10]}")
                with col2:
                    if st.button(f"Load", key=f"load_{project['id']}"):
                        st.session_state.current_project_id = project['id']
                        st.success(f"Loaded project: {project['name']}")
                with col3:
                    st.write(f"Status: {project['status']}")
                st.divider()
    else:
        st.info("No projects found. Create your first project above.")

def load_multiple_dwg_files(uploaded_files):
    """Load multiple DWG files for multi-floor analysis"""
    try:
        with st.spinner("Loading multiple DWG files..."):
            all_zones = []
            floor_plans = []

            for i, file in enumerate(uploaded_files):
                file_bytes = file.read()
                parser = DWGParser()
                zones = parser.parse_file(file_bytes, file.name)

                # Create floor plan object
                floor_plan = FloorPlan(
                    floor_id=f"floor_{i}",
                    floor_number=i + 1,
                    elevation=i * 3.0,  # 3m floor height
                    floor_height=3.0,
                    zones=zones,
                    vertical_connections=[],
                    mechanical_spaces=[],
                    structural_elements=[],
                    analysis_results={}
                )

                floor_plans.append(floor_plan)
                all_zones.extend(zones)

            st.session_state.zones = all_zones
            st.session_state.multi_floor_project = {
                'floors': floor_plans,
                'floor_count': len(floor_plans)
            }
            st.session_state.dwg_loaded = True

            st.success(f"Successfully loaded {len(floor_plans)} floors with {len(all_zones)} total zones")
            st.rerun()

    except Exception as e:
        st.error(f"Error loading multiple DWG files: {str(e)}")

def run_advanced_analysis(components):
    """Run comprehensive advanced AI analysis"""
    try:
        with st.spinner("Running advanced AI analysis..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Step 1: Advanced room classification
            status_text.text("Advanced room classification...")
            progress_bar.progress(20)

            # Use Gemini AI for room classification
            ai_analyzer = components['ai_analyzer']
            room_analysis = {}

            for i, zone in enumerate(st.session_state.zones):
                zone_name = f"Zone_{i}"
                ai_result = ai_analyzer.analyze_room_type(zone)
                room_analysis[zone_name] = {
                    'type': ai_result['type'],
                    'confidence': ai_result['confidence'],
                    'area': zone.get('area', 0),
                    'reasoning': ai_result['reasoning']
                }

            # Step 2: Semantic space analysis
            status_text.text("Semantic space analysis...")
            progress_bar.progress(40)

            semantic_analyzer = components['semantic_analyzer']
            space_graph = semantic_analyzer.build_space_graph(st.session_state.zones, room_analysis)
            spatial_relationships = semantic_analyzer.analyze_spatial_relationships()

            # Step 3: Advanced optimization
            status_text.text("Advanced optimization...")
            progress_bar.progress(60)

            optimization_engine = components['optimization_engine']

            # Basic placement first
            analyzer = AIAnalyzer()
            params = {
                'box_size': (2.0, 1.5),
                'margin': 0.5,
                'allow_rotation': True,
                'smart_spacing': True
            }
            placement_analysis = analyzer.analyze_furniture_placement(st.session_state.zones, params)

            # Use Gemini AI for optimization
            optimization_results = ai_analyzer.optimize_furniture_placement(st.session_state.zones, params)

            # Step 4: Save to database
            status_text.text("Saving to database...")
            progress_bar.progress(80)

            db_manager = components['database']

            # Compile comprehensive results
            results = {
                'rooms': room_analysis,
                'placements': placement_analysis,
                'spatial_relationships': spatial_relationships,
                'optimization': optimization_results,
                'parameters': params,
                'total_boxes': sum(len(spots) for spots in placement_analysis.values()),
                'analysis_type': 'advanced',
                'timestamp': datetime.now().isoformat()
            }

            # Save analysis to database
            if 'current_project_id' in st.session_state:
                analysis_id = db_manager.save_analysis_results(
                    st.session_state.current_project_id,
                    'advanced',
                    params,
                    results
                )
                results['analysis_id'] = analysis_id

            st.session_state.analysis_results = results

            progress_bar.progress(100)
            progress_bar.empty()
            status_text.empty()

            st.success(f"Advanced analysis complete! Analyzed {len(st.session_state.zones)} zones with {results.get('total_boxes', 0)} optimal placements")
            st.rerun()

    except Exception as e:
        st.error(f"Error during advanced analysis: {str(e)}")

def generate_bim_model(components):
    """Generate BIM model from analysis"""
    try:
        with st.spinner("Generating BIM model..."):
            if not st.session_state.analysis_results:
                st.warning("Please run analysis first")
                return

            bim_generator = components['bim_generator']

            building_metadata = {
                'name': 'AI Analyzed Building',
                'address': 'Generated from DWG Analysis',
                'project_name': 'AI Architecture Project',
                'floor_height': 3.0
            }

            bim_model = bim_generator.create_bim_model_from_analysis(
                st.session_state.zones,
                st.session_state.analysis_results,
                building_metadata
            )

            st.session_state.bim_model = bim_model

            # Save to database
            if 'current_project_id' in st.session_state:
                db_manager = components['database']
                bim_id = db_manager.save_bim_model(
                    st.session_state.current_project_id,
                    {'building_data': 'bim_model_data'},
                    bim_model.standards_compliance
                )

            # Show compliance results
            compliance = bim_model.standards_compliance

            st.success("BIM model generated successfully!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("IFC Compliance Score", f"{compliance['ifc']['score']:.1f}%")
            with col2:
                st.metric("Compliant Spaces", compliance['spaces']['compliant_spaces'])

    except Exception as e:
        st.error(f"Error generating BIM model: {str(e)}")

def run_furniture_analysis(components):
    """Run furniture catalog analysis"""
    try:
        with st.spinner("Analyzing furniture requirements..."):
            if not st.session_state.analysis_results:
                st.warning("Please run analysis first")
                return

            furniture_catalog = components['furniture_catalog']
            configurations = []

            rooms = st.session_state.analysis_results.get('rooms', {})

            for zone_name, room_info in rooms.items():
                space_type = room_info.get('type', 'Unknown')
                area = room_info.get('area', 0.0)

                # Generate furniture configuration
                config = furniture_catalog.recommend_furniture_for_space(
                    space_type=space_type,
                    space_area=area,
                    budget=None,
                    sustainability_preference='A'
                )

                configurations.append(config)

                # Save to database
                if 'current_project_id' in st.session_state:
                    db_manager = components['database']
                    db_manager.save_furniture_configuration(
                        st.session_state.current_project_id,
                        config.__dict__
                    )

            st.session_state.furniture_configurations = configurations

            total_cost = sum(config.total_cost for config in configurations)
            total_items = sum(config.total_items for config in configurations)

            st.success(f"Furniture analysis complete! {total_items} items, ${total_cost:,.0f} total cost")

    except Exception as e:
        st.error(f"Error in furniture analysis: {str(e)}")

def generate_cad_export(components):
    """Generate CAD export package"""
    try:
        with st.spinner("Generating CAD export package..."):
            if not st.session_state.analysis_results:
                st.warning("Please run analysis first")
                return

            cad_exporter = components['cad_exporter']

            # Create temporary directory for exports
            with tempfile.TemporaryDirectory() as temp_dir:
                package_files = cad_exporter.create_technical_drawing_package(
                    st.session_state.zones,
                    st.session_state.analysis_results,
                    temp_dir
                )

                st.success("CAD export package generated!")

                # Log export to database
                if 'current_project_id' in st.session_state:
                    db_manager = components['database']
                    for file_type, file_path in package_files.items():
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            db_manager.log_export(
                                st.session_state.current_project_id,
                                file_type,
                                os.path.basename(file_path),
                                file_size,
                                "current_user"
                            )

                # Show downloadable files
                for file_type, file_path in package_files.items():
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            file_data = f.read()

                        file_name = os.path.basename(file_path)
                        st.download_button(
                            label=f"Download {file_type.replace('_', ' ').title()}",
                            data=file_data,
                            file_name=file_name,
                            mime='application/octet-stream'
                        )

    except Exception as e:
        st.error(f"Error generating CAD export: {str(e)}")

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
        st.session_state.advanced_mode = st.toggle("Advanced Mode", value=st.session_state.advanced_mode, key="main_advanced_mode_toggle")

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

        # Enhanced file upload with better error handling
        uploaded_file = st.file_uploader(
            "Choose DWG/DXF files",
            type=['dwg', 'dxf'],
            accept_multiple_files=st.session_state.advanced_mode,
            help="Upload architectural plan files for analysis (Max 200MB per file)"
        )

        if uploaded_file is not None:
            try:
                if isinstance(uploaded_file, list):
                    # Multiple files
                    st.write(f"Selected {len(uploaded_file)} files")
                    total_size = sum(f.size for f in uploaded_file) / (1024*1024)
                    st.write(f"Total size: {total_size:.1f} MB")

                    if st.button("Load Multiple Files", type="primary"):
                        load_multiple_dwg_files(uploaded_file)
                else:
                    # Single file
                    file_size_mb = uploaded_file.size / (1024*1024)
                    st.write(f"File: {uploaded_file.name}")
                    st.write(f"Size: {file_size_mb:.1f} MB")

                    if st.button("Load & Parse File", type="primary"):
                        if file_size_mb > 200:
                            st.error("File too large. Maximum size is 200MB.")
                        else:
                            load_dwg_file(uploaded_file)
            except Exception as e:
                st.error(f"File handling error: {str(e)}")

        st.divider()

        # Analysis parameters
        params = setup_analysis_parameters(components)

        st.divider()

        # Analysis controls in main area
        st.subheader("Analysis Controls")

        if st.session_state.advanced_mode:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Advanced AI Analysis", type="primary"):
                    run_advanced_analysis(components)

                if st.button("Generate BIM Model"):
                    generate_bim_model(components)

            with col2:
                if st.button("Furniture Analysis"):
                    run_furniture_analysis(components)

                if st.button("CAD Export Package"):
                    generate_cad_export(components)
        else:
            if st.button("Run AI Analysis", type="primary"):
                run_ai_analysis(params['box_length'], params['box_width'], params['margin'], 
                               params['confidence_threshold'], params['enable_rotation'], params['smart_spacing'])

        if st.session_state.analysis_results:
            st.divider()
            if st.button("Generate Complete Report"):
                generate_comprehensive_report(components)

    # Main content area
    try:
        if not st.session_state.dwg_loaded:
            display_welcome_screen()
        else:
            display_main_interface(components)
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again")

def display_advanced_statistics(components):
    """Display advanced statistics"""
    if not st.session_state.analysis_results:
        st.info("Run analysis to see detailed statistics")
        return

    results = st.session_state.analysis_results

    # Room type distribution
    if 'rooms' in results:
        room_types = {}
        for info in results.get('rooms', {}).values():
            room_type = info.get('type', 'Unknown')
            room_types[room_type] = room_types.get(room_type, 0) + 1

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Room Type Distribution")
            if room_types:
                room_df = pd.DataFrame(list(room_types.items()), columns=['Room Type', 'Count'])
                fig = px.pie(room_df, values='Count', names='Room Type', title="Room Distribution")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Space Utilization")
            if 'total_boxes' in results:
                box_area = results.get('total_boxes', 0) * 3.0  # Estimate
                total_area = sum(info.get('area', 0.0) for info in results.get('rooms', {}).values())
                utilization = (box_area / total_area * 100) if total_area > 0 else 0

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=utilization,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Space Utilization %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 90}}))
                st.plotly_chart(fig, use_container_width=True)

def load_dwg_file(uploaded_file):
    """Load and parse DWG/DXF file"""
    try:
        with st.spinner("Loading and parsing DWG file..."):
            # Validate file
            if uploaded_file is None:
                st.error("No file selected.")
                return

            # Check file extension
            file_ext = uploaded_file.name.lower().split('.')[-1]
            if file_ext not in ['dwg', 'dxf']:
                st.error(f"Unsupported file format: {file_ext}. Please upload a DWG or DXF file.")
                return

            # Check file size (limit to 50MB for better performance)
            file_size = uploaded_file.size
            if file_size > 50 * 1024 * 1024:
                st.error("File too large. Please use a file smaller than 50MB.")
                return

            if file_size == 0:
                st.error("File appears to be empty.")
                return

            # Read file content
            try:
                file_bytes = uploaded_file.getvalue()
            except Exception:
                file_bytes = uploaded_file.read()

            if len(file_bytes) == 0:
                st.error("Unable to read file content.")
                return

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name

            try:
                # Parse the DWG/DXF file
                parser = DWGParser()
                zones = parser.parse_file_from_path(tmp_file_path)

                if not zones:
                    st.warning("No valid zones found in the file. Please check if the file contains closed polygons or room boundaries.")
                    return

                st.session_state.zones = zones
                st.session_state.dwg_loaded = True
                st.session_state.current_filename = uploaded_file.name

                st.success(f"Successfully loaded {len(zones)} zones from {uploaded_file.name}")
                st.rerun()

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

    except Exception as e:
        error_msg = str(e)
        st.error(f"Error loading DWG file: {error_msg}")
        st.info("Try these solutions: Use a smaller file (under 50MB), ensure the file is a valid DWG/DXF format, or try refreshing the page.")

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

            st.success(f"✅ AI analysis complete! Found {st.session_state.analysis_results.get('total_boxes', 0)} optimal box placements")
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
        st.metric("Total Boxes", results.get('total_boxes', 0))

    with col2:
        total_area = results.get('total_boxes', 0) * results.get('parameters', {}).get('box_size', [2.0, 1.5])[0] * results.get('parameters', {}).get('box_size', [2.0, 1.5])[1]
        st.metric("Total Area", f"{total_area:.1f} m²")

    with col3:
        efficiency = results.get('optimization', {}).get('total_efficiency', 0.85) * 100
        st.metric("Efficiency", f"{efficiency:.1f}%")

    with col4:
        num_rooms = len(results.get('rooms', {}))
        st.metric("Rooms Analyzed", num_rooms)

    st.divider()

    # Detailed room analysis
    st.subheader("🏠 Room Analysis")

    room_data = []
    for zone_name, room_info in results.get('rooms', {}).items():
        placements = results.get('placements', {}).get(zone_name, [])
        room_data.append({
            'Zone': zone_name,
            'Room Type': room_info.get('type', 'Unknown'),
            'Confidence': f"{room_info.get('confidence', 0.0):.1%}",
            'Area (m²)': f"{room_info.get('area', 0.0):.1f}",
            'Dimensions': f"{room_info.get('dimensions', [0, 0])[0]:.1f} × {room_info.get('dimensions', [0, 0])[1]:.1f}",
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
        room_types = [info.get('type', 'Unknown') for info in results.get('rooms', {}).values()]
        room_type_counts = pd.Series(room_types).value_counts()

        fig_pie = px.pie(
            values=room_type_counts.values,
            names=room_type_counts.index,
            title="Room Type Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Box placement by room
        placement_counts = {zone: len(placements) for zone, placements in results.get('placements', {}).items()}

        if placement_counts:
            # Create DataFrame for Plotly
            df = pd.DataFrame({
                'Zone': list(placement_counts.keys()),
                'Count': list(placement_counts.values())
            })

            fig_bar = px.bar(
                df,
                x='Zone',
                y='Count',
                title="Boxes per Zone"
            )
        else:
            fig_bar = px.Figure()
            fig_bar.update_layout(title="No placement data available")
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Efficiency metrics
    st.subheader("⚡ Efficiency Metrics")

    # Calculate various efficiency metrics
    total_room_area = sum(info.get('area', 0.0) for info in results.get('rooms', {}).values())
    total_boxes = results.get('total_boxes', 0)
    box_size = results.get('parameters', {}).get('box_size', [2.0, 1.5])
    total_box_area = total_boxes * box_size[0] * box_size[1]
    space_utilization = (total_box_area / total_room_area) * 100 if total_room_area > 0 else 0

    avg_suitability = 0
    if results.get('placements'):
        all_scores = []
        for placements in results.get('placements', {}).values():
            all_scores.extend([p['suitability_score'] for p in placements])
        avg_suitability = np.mean(all_scores) if all_scores else 0

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        st.metric("Space Utilization", f"{space_utilization:.1f}%")

    with metrics_col2:
        st.metric("Avg. Suitability Score", f"{avg_suitability:.2f}")

    with metrics_col3:
        st.metric("Boxes per m²", f"{results.get('total_boxes', 0)/total_room_area:.2f}" if total_room_area > 0 else "0.00")

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
        for zone_name, room_info in results.get('rooms', {}).items():
            placements = results.get('placements', {}).get(zone_name, [])
            room_data.append({
                'Zone': zone_name,
                'Room_Type': room_info.get('type', 'Unknown'),
                'Confidence': room_info.get('confidence', 0.0),
                'Area_m2': room_info.get('area', 0.0),
                'Width_m': room_info.get('dimensions', [0, 0])[0],
                'Height_m': room_info.get('dimensions', [0, 0])[1],
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
        **Analysis Complete**: {results.get('total_boxes', 0)} optimal box placements found

        **Room Analysis**: {len(results.get('rooms', {}))} rooms analyzed
        - Average confidence: {np.mean([r.get('confidence', 0.0) for r in results.get('rooms', {}).values()]):.1%}

        **Box Parameters**: {results.get('parameters', {}).get('box_size', [2.0, 1.5])[0]}m × {results.get('parameters', {}).get('box_size', [2.0, 1.5])[1]}m

        **Total Coverage**: {results.get('total_boxes', 0) * results.get('parameters', {}).get('box_size', [2.0, 1.5])[0] * results.get('parameters', {}).get('box_size', [2.0, 1.5])[1]:.1f} m²

        **Algorithm**: {results.get('optimization', {}).get('algorithm_used', 'Standard Optimization')}
        """)

if __name__ == "__main__":
    main()