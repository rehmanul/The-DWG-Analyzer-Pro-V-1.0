import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

class PlanVisualizer:
    """Visualization class for architectural plans"""
    
    def create_basic_plot(self, zones):
        """Create basic 2D plot of zones"""
        fig = go.Figure()
        
        for zone in zones:
            # Add zone outline
            fig.add_trace(go.Scatter(
                x=zone.get('points', [[]])[0],
                y=zone.get('points', [[]])[1],
                fill="toself",
                mode='lines',
                name=f"Zone {zone.get('id', 'Unknown')}"
            ))
        
        fig.update_layout(
            title="Floor Plan Visualization",
            showlegend=True,
            hovermode='closest'
        )
        
        return fig

    def create_interactive_plot(self, zones, analysis_results, show_zones=True, 
                              show_boxes=True, show_labels=True, color_by_type=True):
        """Create interactive plot with analysis results"""
        fig = go.Figure()
        
        if show_zones:
            for zone in zones:
                fig.add_trace(go.Scatter(
                    x=zone.get('points', [[]])[0],
                    y=zone.get('points', [[]])[1],
                    fill="toself",
                    mode='lines+text' if show_labels else 'lines',
                    name=f"Zone {zone.get('id', 'Unknown')}",
                    text=analysis_results.get('rooms', {}).get(str(zone.get('id')), {}).get('type', 'Unknown'),
                    textposition="middle center"
                ))
        
        fig.update_layout(
            title="Interactive Floor Plan Analysis",
            showlegend=True,
            hovermode='closest',
            height=600
        )
        
        return fig

    def create_3d_plot(self, zones, analysis_results):
        """Create 3D visualization"""
        fig = go.Figure()
        
        # Base height for 3D view
        base_height = 3.0
        
        for zone in zones:
            # Create 3D surface
            x = zone.get('points', [[]])[0]
            y = zone.get('points', [[]])[1]
            z = [base_height] * len(x)
            
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                opacity=0.8,
                name=f"Zone {zone.get('id', 'Unknown')}"
            ))
        
        fig.update_layout(
            title="3D Floor Plan Visualization",
            scene=dict(
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600
        )
        
        return fig

    def display_statistics(self, results):
        """Display detailed statistics with fixed Plotly configuration"""
        if not results:
            st.info("Run AI analysis to see statistics")
            return

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
                fig_bar = go.Figure(data=[
                    go.Bar(x=list(placement_counts.keys()), y=list(placement_counts.values()))
                ])
                fig_bar.update_layout(
                    title="Boxes per Zone",
                    xaxis=dict(title="Zone", tickangle=45),
                    yaxis=dict(title="Number of Boxes"),
                    margin=dict(t=50, l=50, r=50, b=100),
                    height=400
                )
            else:
                fig_bar = go.Figure()
                fig_bar.update_layout(
                    title="No placement data available",
                    xaxis=dict(title="Zone", tickangle=45),
                    yaxis=dict(title="Number of Boxes"),
                    margin=dict(t=50, l=50, r=50, b=100),
                    height=400
                )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Efficiency metrics
        st.subheader("⚡ Efficiency Metrics")

        # Calculate metrics
        total_room_area = sum(info.get('area', 0.0) for info in results.get('rooms', {}).values())
        total_boxes = results.get('total_boxes', 0)
        box_size = results.get('parameters', {}).get('box_size', [2.0, 1.5])
        total_box_area = total_boxes * box_size[0] * box_size[1]
        space_utilization = (total_box_area / total_room_area) * 100 if total_room_area > 0 else 0

        # Display metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            st.metric("Space Utilization", f"{space_utilization:.1f}%")

        with metrics_col2:
            avg_suitability = 0
            if results.get('placements'):
                all_scores = []
                for placements in results.get('placements', {}).values():
                    all_scores.extend([p.get('suitability_score', 0) for p in placements])
                avg_suitability = sum(all_scores) / len(all_scores) if all_scores else 0
            st.metric("Avg. Suitability Score", f"{avg_suitability:.2f}")

        with metrics_col3:
            boxes_per_m2 = results.get('total_boxes', 0) / total_room_area if total_room_area > 0 else 0
            st.metric("Boxes per m²", f"{boxes_per_m2:.2f}")
