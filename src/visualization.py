import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional

class PlanVisualizer:
    """
    Visualization utilities for architectural plans and analysis results
    """
    
    def __init__(self):
        self.room_colors = {
            'Office': '#3498db',
            'Small Office': '#5dade2', 
            'Open Office': '#2980b9',
            'Meeting Room': '#e74c3c',
            'Conference Room': '#c0392b',
            'Corridor': '#f39c12',
            'Storage/WC': '#95a5a6',
            'Hall/Auditorium': '#9b59b6',
            'Living Room': '#27ae60',
            'Bedroom': '#16a085',
            'Kitchen': '#e67e22',
            'Bathroom': '#34495e',
            'Unknown': '#bdc3c7'
        }
        
        self.box_color = '#2ecc71'
        self.box_alpha = 0.7
    
    def create_interactive_plot(self, zones: List[Dict], analysis_results: Dict,
                              show_zones: bool = True, show_boxes: bool = True,
                              show_labels: bool = True, color_by_type: bool = True) -> go.Figure:
        """
        Create interactive Plotly visualization of the architectural plan
        """
        fig = go.Figure()
        
        if show_zones and zones:
            self._add_zones_to_plot(fig, zones, analysis_results, color_by_type, show_labels)
        
        if show_boxes and analysis_results.get('placements'):
            self._add_boxes_to_plot(fig, analysis_results['placements'])
        
        # Update layout
        fig.update_layout(
            title="Architectural Plan Analysis",
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            showlegend=True,
            hovermode='closest',
            width=800,
            height=600,
            xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal aspect ratio
        )
        
        return fig
    
    def create_basic_plot(self, zones: List[Dict]) -> go.Figure:
        """
        Create basic plot showing only zones without analysis
        """
        fig = go.Figure()
        
        for i, zone in enumerate(zones):
            if not zone.get('points'):
                continue
            
            points = zone['points']
            x_coords = [p[0] for p in points] + [points[0][0]]  # Close polygon
            y_coords = [p[1] for p in points] + [points[0][1]]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                name=f"Zone {i}",
                line=dict(color='blue', width=2),
                hovertemplate=f"Zone {i}<br>Layer: {zone.get('layer', 'Unknown')}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Architectural Plan - Zones Only",
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            showlegend=True,
            xaxis=dict(scaleanchor="y", scaleratio=1),
        )
        
        return fig
    
    def create_3d_plot(self, zones: List[Dict], analysis_results: Dict) -> go.Figure:
        """
        Create 3D visualization of box placements
        """
        fig = go.Figure()
        
        if not analysis_results.get('placements'):
            return fig
        
        box_height = 2.0  # Standard box height for 3D visualization
        
        for zone_name, placements in analysis_results['placements'].items():
            for i, placement in enumerate(placements):
                # Get box coordinates
                coords = placement['box_coords']
                x_coords = [p[0] for p in coords]
                y_coords = [p[1] for p in coords]
                
                # Create 3D box
                self._add_3d_box(fig, x_coords, y_coords, box_height, 
                                f"{zone_name}_Box_{i+1}")
        
        fig.update_layout(
            title="3D Box Placement Visualization",
            scene=dict(
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)",
                zaxis_title="Z (meters)",
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.3)
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def _add_zones_to_plot(self, fig: go.Figure, zones: List[Dict], 
                          analysis_results: Dict, color_by_type: bool, show_labels: bool):
        """Add zone polygons to the plot"""
        rooms = analysis_results.get('rooms', {})
        
        for i, zone in enumerate(zones):
            if not zone.get('points'):
                continue
            
            zone_name = f"Zone_{i}"
            points = zone['points']
            x_coords = [p[0] for p in points] + [points[0][0]]
            y_coords = [p[1] for p in points] + [points[0][1]]
            
            # Determine color
            if color_by_type and zone_name in rooms:
                room_type = rooms[zone_name]['type']
                color = self.room_colors.get(room_type, self.room_colors['Unknown'])
                confidence = rooms[zone_name]['confidence']
                alpha = max(0.3, confidence)  # Use confidence for alpha
            else:
                color = 'blue'
                alpha = 0.3
            
            # Create hover text
            hover_text = f"Zone {i}<br>Layer: {zone.get('layer', 'Unknown')}"
            if zone_name in rooms:
                room_info = rooms[zone_name]
                hover_text += f"<br>Type: {room_info['type']}"
                hover_text += f"<br>Confidence: {room_info['confidence']:.1%}"
                hover_text += f"<br>Area: {room_info['area']:.1f} m²"
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                name=rooms[zone_name]['type'] if zone_name in rooms else f"Zone {i}",
                line=dict(color=color, width=2),
                fill='toself',
                fillcolor=f"rgba{self._hex_to_rgba(color, alpha)}",
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=True
            ))
            
            # Add labels if requested
            if show_labels and zone_name in rooms:
                # Calculate centroid for label placement
                centroid_x = sum(p[0] for p in points) / len(points)
                centroid_y = sum(p[1] for p in points) / len(points)
                
                fig.add_annotation(
                    x=centroid_x,
                    y=centroid_y,
                    text=rooms[zone_name]['type'],
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1
                )
    
    def _add_boxes_to_plot(self, fig: go.Figure, placements: Dict[str, List[Dict]]):
        """Add box placements to the plot"""
        box_count = 0
        
        for zone_name, zone_placements in placements.items():
            for placement in zone_placements:
                coords = placement['box_coords']
                x_coords = [p[0] for p in coords] + [coords[0][0]]
                y_coords = [p[1] for p in coords] + [coords[0][1]]
                
                box_count += 1
                
                # Color intensity based on suitability score
                score = placement['suitability_score']
                alpha = 0.4 + (score * 0.4)  # Range from 0.4 to 0.8
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    name=f"Box {box_count}" if box_count <= 10 else "Boxes",
                    line=dict(color=self.box_color, width=1),
                    fill='toself',
                    fillcolor=f"rgba{self._hex_to_rgba(self.box_color, alpha)}",
                    hovertemplate=(
                        f"Box {box_count}<br>"
                        f"Size: {placement['size'][0]:.1f}×{placement['size'][1]:.1f}m<br>"
                        f"Suitability: {score:.2f}<br>"
                        f"Area: {placement['area']:.1f} m²<extra></extra>"
                    ),
                    showlegend=box_count <= 10,  # Only show legend for first 10 boxes
                    legendgroup="boxes"
                ))
    
    def _add_3d_box(self, fig: go.Figure, x_coords: List[float], y_coords: List[float], 
                   height: float, name: str):
        """Add a 3D box to the figure"""
        # Bottom face
        fig.add_trace(go.Mesh3d(
            x=x_coords + x_coords,
            y=y_coords + y_coords,
            z=[0] * len(x_coords) + [height] * len(x_coords),
            i=[0, 1, 2, 3, 4, 5, 6, 7],
            j=[1, 2, 3, 0, 5, 6, 7, 4],
            k=[5, 6, 7, 4, 1, 2, 3, 0],
            opacity=0.7,
            color=self.box_color,
            name=name,
            showlegend=False
        ))
    
    def _hex_to_rgba(self, hex_color: str, alpha: float) -> str:
        """Convert hex color to RGBA string"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"
    
    def create_statistics_charts(self, analysis_results: Dict) -> Dict[str, go.Figure]:
        """Create various statistical charts"""
        charts = {}
        
        if not analysis_results.get('rooms'):
            return charts
        
        rooms = analysis_results['rooms']
        placements = analysis_results.get('placements', {})
        
        # Room type distribution pie chart
        room_types = [info['type'] for info in rooms.values()]
        room_type_counts = {}
        for room_type in room_types:
            room_type_counts[room_type] = room_type_counts.get(room_type, 0) + 1
        
        charts['room_distribution'] = px.pie(
            values=list(room_type_counts.values()),
            names=list(room_type_counts.keys()),
            title="Room Type Distribution",
            color_discrete_map=self.room_colors
        )
        
        # Box placement per zone bar chart
        placement_counts = {zone: len(zone_placements) 
                          for zone, zone_placements in placements.items()}
        
        charts['boxes_per_zone'] = px.bar(
            x=list(placement_counts.keys()),
            y=list(placement_counts.values()),
            title="Boxes per Zone",
            labels={'x': 'Zone', 'y': 'Number of Boxes'}
        )
        
        # Area utilization chart
        zone_areas = []
        zone_names = []
        utilized_areas = []
        
        for zone_name, room_info in rooms.items():
            zone_areas.append(room_info['area'])
            zone_names.append(zone_name)
            
            # Calculate utilized area
            zone_placements = placements.get(zone_name, [])
            utilized_area = sum(p['area'] for p in zone_placements)
            utilized_areas.append(utilized_area)
        
        charts['area_utilization'] = go.Figure(data=[
            go.Bar(name='Total Area', x=zone_names, y=zone_areas, opacity=0.7),
            go.Bar(name='Utilized Area', x=zone_names, y=utilized_areas, opacity=0.9)
        ])
        charts['area_utilization'].update_layout(
            title='Area Utilization by Zone',
            xaxis_title='Zone',
            yaxis_title='Area (m²)',
            barmode='overlay'
        )
        
        return charts
