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
            fig.add_trace(
                go.Scatter(x=zone.get('points', [[]])[0],
                           y=zone.get('points', [[]])[1],
                           fill="toself",
                           mode='lines',
                           name=f"Zone {zone.get('id', 'Unknown')}"))

        fig.update_layout(title="Floor Plan Visualization",
                          showlegend=True,
                          hovermode='closest')

        return fig

    def create_3d_plot(self, zones: List[Dict],
                       analysis_results: Dict) -> go.Figure:
        """
        Create 3D visualization of box placements and zones
        """
        fig = go.Figure()

        # Add zones as 3D base
        for i, zone in enumerate(zones):
            if zone.get('points') and len(zone['points']) >= 3:
                points = zone['points']
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                z_coords = [0] * len(points)

                # Add floor plane
                fig.add_trace(
                    go.Scatter3d(x=x_coords + [x_coords[0]],
                                 y=y_coords + [y_coords[0]],
                                 z=z_coords + [z_coords[0]],
                                 mode='lines',
                                 name=f'Zone {i} Floor',
                                 line=dict(color='blue', width=3),
                                 showlegend=True))

                # Add zone walls (extruded to height)
                wall_height = 3.0
                for j in range(len(points)):
                    next_j = (j + 1) % len(points)
                    fig.add_trace(
                        go.Scatter3d(x=[
                            x_coords[j], x_coords[j], x_coords[next_j],
                            x_coords[next_j], x_coords[j]
                        ],
                                     y=[
                                         y_coords[j], y_coords[j],
                                         y_coords[next_j], y_coords[next_j],
                                         y_coords[j]
                                     ],
                                     z=[0, wall_height, wall_height, 0, 0],
                                     mode='lines',
                                     line=dict(color='lightblue', width=1),
                                     showlegend=False))

        # Add box placements if available
        if analysis_results.get('placements'):
            box_height = 2.0
            box_count = 0

            for zone_name, placements in analysis_results['placements'].items(
            ):
                for placement in placements:
                    box_count += 1
                    if 'box_coords' in placement:
                        coords = placement['box_coords']
                        x_coords = [p[0] for p in coords]
                        y_coords = [p[1] for p in coords]
                        self._add_3d_box(fig, x_coords, y_coords, box_height,
                                         f"Box_{box_count}")
                    elif 'position' in placement and 'size' in placement:
                        # Alternative format
                        pos = placement['position']
                        size = placement['size']
                        x, y = pos[0], pos[1]
                        w, h = size[0], size[1]

                        x_coords = [x - w / 2, x + w / 2, x + w / 2, x - w / 2]
                        y_coords = [y - h / 2, y - h / 2, y + h / 2, y + h / 2]
                        self._add_3d_box(fig, x_coords, y_coords, box_height,
                                         f"Box_{box_count}")

        fig.update_layout(title="3D Architectural Visualization",
                          scene=dict(xaxis_title="X (meters)",
                                     yaxis_title="Y (meters)",
                                     zaxis_title="Z (meters)",
                                     aspectmode='cube'),
                          width=900,
                          height=700)

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

            fig.add_trace(
                go.Mesh3d(x=x,
                          y=y,
                          z=z,
                          opacity=0.8,
                          name=f"Zone {zone.get('id', 'Unknown')}"))

        fig.update_layout(title="3D Floor Plan Visualization",
                          scene=dict(aspectmode='data',
                                     camera=dict(up=dict(x=0, y=0, z=1),
                                                 center=dict(x=0, y=0, z=0),
                                                 eye=dict(x=1.5, y=1.5,
                                                          z=1.5))),
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
            room_type_counts = pd.Series(room_types).value_counts()

            fig_pie = px.pie(values=room_type_counts.values,
                             names=room_type_counts.index,
                             title="Room Type Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

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
            else:
                fig_bar = go.Figure()
                fig_bar.update_layout(title="No placement data available",
                                      xaxis=dict(title="Zone", tickangle=45),
                                      yaxis=dict(title="Number of Boxes"),
                                      margin=dict(t=50, l=50, r=50, b=100),
                                      height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Efficiency metrics
        st.subheader("⚡ Efficiency Metrics")

        # Calculate metrics
        total_room_area = sum(
            info.get('area', 0.0)
            for info in results.get('rooms', {}).values())
        total_boxes = results.get('total_boxes', 0)
        box_size = results.get('parameters', {}).get('box_size', [2.0, 1.5])
        total_box_area = total_boxes * box_size[0] * box_size[1]
        space_utilization = (total_box_area / total_room_area
                             ) * 100 if total_room_area > 0 else 0

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
                avg_suitability = sum(all_scores) / len(
                    all_scores) if all_scores else 0
            st.metric("Avg. Suitability Score", f"{avg_suitability:.2f}")

        with metrics_col3:
            boxes_per_m2 = results.get(
                'total_boxes',
                0) / total_room_area if total_room_area > 0 else 0
            st.metric("Boxes per m²", f"{boxes_per_m2:.2f}")
"""
Basic visualization module for DWG analysis
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Dict, Any

class PlanVisualizer:
    """Basic plan visualization class"""
    
    def create_basic_plot(self, zones: List[Dict]) -> go.Figure:
        """Create basic zone visualization"""
        fig = go.Figure()
        
        for i, zone in enumerate(zones):
            if 'points' in zone and zone['points']:
                points = zone['points']
                # Close the polygon
                x_coords = [p[0] for p in points] + [points[0][0]]
                y_coords = [p[1] for p in points] + [points[0][1]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    fill='toself',
                    name=f"Zone {i}",
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="DWG Zone Analysis",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig
    
    def create_interactive_plot(self, zones: List[Dict], analysis_results: Dict, 
                              show_zones: bool = True, show_boxes: bool = True,
                              show_labels: bool = True, color_by_type: bool = True) -> go.Figure:
        """Create interactive plot with analysis results"""
        fig = self.create_basic_plot(zones)
        
        if analysis_results and 'placements' in analysis_results:
            # Add furniture placements
            for zone_name, placements in analysis_results['placements'].items():
                for placement in placements:
                    x, y = placement['position']
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='square'),
                        name='Furniture',
                        showlegend=False
                    ))
        
        return fig
    
    def create_3d_plot(self, zones: List[Dict], analysis_results: Dict) -> go.Figure:
        """Create 3D visualization"""
        fig = go.Figure()
        
        for i, zone in enumerate(zones):
            if 'points' in zone and zone['points']:
                points = zone['points']
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                z_coords = [0] * len(points)  # Ground level
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers+lines',
                    name=f"Zone {i}"
                ))
        
        fig.update_layout(
            title="3D Plan View",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            )
        )
        
        return fig
