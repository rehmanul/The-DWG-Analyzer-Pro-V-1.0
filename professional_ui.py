"""
Professional UI Components for AI Architectural Space Analyzer PRO
Enhanced production-ready interface components
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

class ProfessionalUI:
    """Professional UI components and styling"""
    
    @staticmethod
    def render_header():
        """Render professional application header"""
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%); 
                    padding: 2rem; margin: -1rem -1rem 2rem -1rem; color: white;">
            <h1 style="margin: 0; font-size: 2.5rem; font-weight: 300;">
                🏗️ AI Architectural Space Analyzer PRO
            </h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                Enterprise-grade DWG/DXF analysis with AI-powered insights
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_stats_dashboard(zones, analysis_results):
        """Render professional statistics dashboard"""
        if not zones:
            return
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Zones Detected",
                value=len(zones),
                delta=f"+{len(zones)} from analysis"
            )
        
        with col2:
            total_area = sum(zone.get('area', 0) for zone in zones)
            st.metric(
                label="Total Area (sq ft)",
                value=f"{total_area:,.0f}",
                delta="Calculated from CAD"
            )
        
        with col3:
            room_types = len(set(result.get('room_type', 'Unknown') 
                               for result in analysis_results.values()))
            st.metric(
                label="Room Types Identified",
                value=room_types,
                delta="AI Classification"
            )
        
        with col4:
            avg_confidence = sum(result.get('confidence', 0) 
                               for result in analysis_results.values()) / max(len(analysis_results), 1)
            st.metric(
                label="Avg. AI Confidence",
                value=f"{avg_confidence:.1%}",
                delta="Machine Learning"
            )
    
    @staticmethod
    def render_professional_sidebar():
        """Render enhanced professional sidebar"""
        with st.sidebar:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: #333;">Professional Features</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666;">
                    Enterprise-grade architectural analysis
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature toggles
            st.subheader("Analysis Features")
            
            ai_analysis = st.checkbox("AI Room Classification", value=True, 
                                    help="Use advanced AI for room type detection")
            
            bim_compliance = st.checkbox("BIM Compliance Check", value=True,
                                       help="Validate against IFC 4.0 standards")
            
            optimization = st.checkbox("Space Optimization", value=True,
                                     help="Generate furniture placement recommendations")
            
            collaboration = st.checkbox("Team Collaboration", value=False,
                                      help="Enable real-time collaborative features")
            
            # Export options
            st.subheader("Export Options")
            
            export_formats = st.multiselect(
                "Select Export Formats",
                ["PDF Report", "DXF CAD", "SVG Vector", "JSON Data", "CSV Analytics"],
                default=["PDF Report", "JSON Data"]
            )
            
            return {
                'ai_analysis': ai_analysis,
                'bim_compliance': bim_compliance,
                'optimization': optimization,
                'collaboration': collaboration,
                'export_formats': export_formats
            }
    
    @staticmethod
    def render_progress_indicator(step, total_steps, current_task="Processing"):
        """Render professional progress indicator"""
        progress = step / total_steps
        
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong>{current_task}</strong>
                <span>{step}/{total_steps}</span>
            </div>
            <div style="background: #e9ecef; height: 8px; border-radius: 4px; margin-top: 0.5rem;">
                <div style="background: #1f77b4; height: 100%; width: {progress*100}%; 
                           border-radius: 4px; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_error_handler(error_type, error_message, solutions=None):
        """Render professional error display with solutions"""
        st.error(f"**{error_type}**: {error_message}")
        
        if solutions:
            with st.expander("Suggested Solutions"):
                for i, solution in enumerate(solutions, 1):
                    st.write(f"{i}. {solution}")
    
    @staticmethod
    def render_feature_showcase():
        """Render professional feature showcase"""
        st.markdown("""
        ### 🚀 Professional Features Available
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 1rem; margin: 1rem 0;">
            
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1f77b4;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1f77b4;">AI-Powered Analysis</h4>
                <p style="margin: 0; color: #666;">Advanced machine learning for 95%+ accurate room classification</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2ca02c;">
                <h4 style="margin: 0 0 0.5rem 0; color: #2ca02c;">BIM Integration</h4>
                <p style="margin: 0; color: #666;">IFC 4.0 compliance checking and professional CAD export</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #ff7f0e;">
                <h4 style="margin: 0 0 0.5rem 0; color: #ff7f0e;">Space Optimization</h4>
                <p style="margin: 0; color: #666;">Genetic algorithms for optimal furniture placement</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #d62728;">
                <h4 style="margin: 0 0 0.5rem 0; color: #d62728;">Team Collaboration</h4>
                <p style="margin: 0; color: #666;">Real-time commenting and project management</p>
            </div>
            
        </div>
        """, unsafe_allow_html=True)

class DataVisualization:
    """Professional data visualization components"""
    
    @staticmethod
    def create_room_analysis_chart(analysis_results):
        """Create professional room analysis chart"""
        if not analysis_results:
            return None
            
        room_types = {}
        confidences = []
        
        for result in analysis_results.values():
            room_type = result.get('room_type', 'Unknown')
            confidence = result.get('confidence', 0)
            
            room_types[room_type] = room_types.get(room_type, 0) + 1
            confidences.append(confidence)
        
        # Create pie chart for room distribution
        fig = go.Figure(data=[
            go.Pie(
                labels=list(room_types.keys()),
                values=list(room_types.values()),
                hole=0.4,
                textinfo="label+percent",
                textposition="outside",
                marker=dict(
                    colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
                    line=dict(color='#FFFFFF', width=2)
                )
            )
        ])
        
        fig.update_layout(
            title={
                'text': "Room Type Distribution",
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            font=dict(size=12),
            showlegend=True,
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    @staticmethod
    def create_confidence_chart(analysis_results):
        """Create AI confidence analysis chart"""
        if not analysis_results:
            return None
            
        confidences = [result.get('confidence', 0) for result in analysis_results.values()]
        room_types = [result.get('room_type', 'Unknown') for result in analysis_results.values()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=room_types,
                y=confidences,
                marker=dict(
                    color=confidences,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Confidence Level")
                ),
                text=[f"{c:.1%}" for c in confidences],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title={
                'text': "AI Classification Confidence",
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Room Type",
            yaxis_title="Confidence Level",
            yaxis=dict(tickformat='.0%'),
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig