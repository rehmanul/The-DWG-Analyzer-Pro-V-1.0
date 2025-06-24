import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class PlanVisualizer:
    """Visualization class for architectural plans"""

    def __init__(self):
        pass

    def create_basic_plot(self, zones):
        """Create basic 2D plot of zones"""
        fig = go.Figure()

        for i, zone in enumerate(zones):
            if 'points' in zone and zone['points']:
                points = zone['points']
                # Handle different point formats
                if isinstance(points[0], (list, tuple)):
                    x_coords = [p[0] for p in points] + [points[0][0]]
                    y_coords = [p[1] for p in points] + [points[0][1]]
                else:
                    # Assume points is already in [x_coords, y_coords] format
                    x_coords = points[0] if len(points) > 0 else []
                    y_coords = points[1] if len(points) > 1 else []

                fig.add_trace(
                    go.Scatter(x=x_coords,
                               y=y_coords,
                               fill="toself",
                               mode='lines',
                               name=f"Zone {zone.get('id', i)}",
                               line=dict(width=2)))

        fig.update_layout(title="Floor Plan Visualization",
                          xaxis_title="X Coordinate",
                          yaxis_title="Y Coordinate",
                          showlegend=True,
                          hovermode='closest',
                          width=800,
                          height=600)

        return fig

    def create_interactive_plot(self, zones, analysis_results, show_zones=True, 
                              show_boxes=True, show_labels=True, color_by_type=True):
        """Create interactive plot with analysis results"""
        fig = go.Figure()

        if show_zones:
            for i, zone in enumerate(zones):
                if 'points' in zone and zone['points']:
                    points = zone['points']
                    # Handle different point formats
                    if isinstance(points[0], (list, tuple)):
                        x_coords = [p[0] for p in points] + [points[0][0]]
                        y_coords = [p[1] for p in points] + [points[0][1]]
                    else:
                        x_coords = points[0] if len(points) > 0 else []
                        y_coords = points[1] if len(points) > 1 else []

                    # Color by room type if analysis available
                    color = 'blue'
                    if color_by_type and analysis_results and 'rooms' in analysis_results:
                        zone_key = f"zone_{i}"
                        if zone_key in analysis_results['rooms']:
                            room_type = analysis_results['rooms'][zone_key].get('type', 'Unknown')
                            color_map = {
                                'Office': 'lightblue',
                                'Conference Room': 'lightgreen', 
                                'Kitchen': 'orange',
                                'Bathroom': 'pink',
                                'Storage': 'gray',
                                'Unknown': 'lightgray'
                            }
                            color = color_map.get(room_type, 'lightgray')

                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        fill="toself",
                        mode='lines+text' if show_labels else 'lines',
                        name=f"Zone {zone.get('id', i)}",
                        fillcolor=color,
                        line=dict(width=2, color='black'),
                        text=f"Zone {i}" if show_labels else None,
                        textposition="middle center"
                    ))

        # Add box placements if available
        if show_boxes and analysis_results and 'placements' in analysis_results:
            for zone_name, placements in analysis_results['placements'].items():
                for j, placement in enumerate(placements):
                    if 'position' in placement:
                        x, y = placement['position']
                        fig.add_trace(go.Scatter(
                            x=[x], y=[y],
                            mode='markers',
                            marker=dict(size=12, color='red', symbol='square'),
                            name='Furniture' if j == 0 else None,
                            showlegend=(j == 0),
                            hovertemplate=f"<b>Furniture Placement</b><br>Position: ({x:.1f}, {y:.1f})<br>Zone: {zone_name}<extra></extra>"
                        ))

        fig.update_layout(
            title="Interactive Floor Plan Analysis",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate", 
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

        for i, zone in enumerate(zones):
            if 'points' in zone and zone['points']:
                points = zone['points']
                if isinstance(points[0], (list, tuple)):
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                else:
                    x_coords = points[0] if len(points) > 0 else []
                    y_coords = points[1] if len(points) > 1 else []

                z_coords = [0] * len(x_coords)
                z_top = [base_height] * len(x_coords)

                # Add floor
                fig.add_trace(
                    go.Scatter3d(x=x_coords + [x_coords[0]] if x_coords else [],
                                 y=y_coords + [y_coords[0]] if y_coords else [],
                                 z=z_coords + [z_coords[0]] if z_coords else [],
                                 mode='lines',
                                 name=f'Zone {i} Floor',
                                 line=dict(color='blue', width=3)))

                # Add ceiling
                fig.add_trace(
                    go.Scatter3d(x=x_coords + [x_coords[0]] if x_coords else [],
                                 y=y_coords + [y_coords[0]] if y_coords else [],
                                 z=z_top + [z_top[0]] if z_top else [],
                                 mode='lines',
                                 name=f'Zone {i} Ceiling',
                                 line=dict(color='lightblue', width=2)))

        # Add furniture placements in 3D
        if analysis_results and 'placements' in analysis_results:
            for zone_name, placements in analysis_results['placements'].items():
                for placement in placements:
                    if 'position' in placement:
                        x, y = placement['position']
                        fig.add_trace(go.Scatter3d(
                            x=[x], y=[y], z=[1.0],  # 1m height for furniture
                            mode='markers',
                            marker=dict(size=8, color='red', symbol='cube'),
                            name='Furniture',
                            showlegend=False
                        ))

        fig.update_layout(title="3D Floor Plan Visualization",
                          scene=dict(aspectmode='data',
                                     camera=dict(up=dict(x=0, y=0, z=1),
                                                 center=dict(x=0, y=0, z=0),
                                                 eye=dict(x=1.5, y=1.5, z=1.5)),
                                     xaxis_title="X (meters)",
                                     yaxis_title="Y (meters)",
                                     zaxis_title="Z (meters)"),
                          height=600)

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
            room_types = [
                info.get('type', 'Unknown')
                for info in results.get('rooms', {}).values()
            ]
            if room_types:
                room_type_counts = pd.Series(room_types).value_counts()

                fig_pie = px.pie(values=room_type_counts.values,
                                 names=room_type_counts.index,
                                 title="Room Type Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No room type data available")

        with col2:
            # Box placement by room
            placement_counts = {
                zone: len(placements)
                for zone, placements in results.get('placements', {}).items()
            }

            if placement_counts:
                fig_bar = go.Figure(data=[
                    go.Bar(x=list(placement_counts.keys()),
                           y=list(placement_counts.values()))
                ])
                fig_bar.update_layout(title="Boxes per Zone",
                                      xaxis=dict(title="Zone", tickangle=45),
                                      yaxis=dict(title="Number of Boxes"),
                                      margin=dict(t=50, l=50, r=50, b=100),
                                      height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No placement data available")

        # Efficiency metrics
        st.subheader("⚡ Efficiency Metrics")

        # Calculate metrics
        total_room_area = sum(
            info.get('area', 0.0)
            for info in results.get('rooms', {}).values())
        total_boxes = results.get('total_boxes', 0)
        box_size = results.get('parameters', {}).get('box_size', [2.0, 1.5])
        total_box_area = total_boxes * box_size[0] * box_size[1] if isinstance(box_size, (list, tuple)) else 0
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
                    all_scores.extend(
                        [p.get('suitability_score', 0) for p in placements])
                avg_suitability = sum(all_scores) / len(all_scores) if all_scores else 0
            st.metric("Avg. Suitability Score", f"{avg_suitability:.2f}")

        with metrics_col3:
            boxes_per_m2 = total_boxes / total_room_area if total_room_area > 0 else 0
            st.metric("Boxes per m²", f"{boxes_per_m2:.2f}")