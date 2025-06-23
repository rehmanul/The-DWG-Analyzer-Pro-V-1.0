import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

class PlanVisualizer:
    def __init__(self):
        pass

    def create_basic_plot(self, zones):
        fig = go.Figure()
        for zone in zones:
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
        fig = go.Figure()
        if show_zones:
            for zone in zones:
                fig.add_trace(go.Scatter(
                    x=zone.get('points', [[]])[0],
                    y=zone.get('points', [[]])[1],
                    fill="toself",
                    mode='lines+text' if show_labels else 'lines',
                    name=f"Zone {zone.get('id', 'Unknown')}"
                ))
        fig.update_layout(
            title="Interactive Floor Plan Analysis",
            showlegend=True,
            hovermode='closest',
            height=600
        )
        return fig

    def create_3d_plot(self, zones, analysis_results):
        fig = go.Figure()
        for zone in zones:
            x = zone.get('points', [[]])[0]
            y = zone.get('points', [[]])[1]
            z = [3.0] * len(x)
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
