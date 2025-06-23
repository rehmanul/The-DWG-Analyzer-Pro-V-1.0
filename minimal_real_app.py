import streamlit as st
import ezdxf
import tempfile
import os
import plotly.graph_objects as go
from shapely.geometry import Polygon

st.set_page_config(
    page_title="Real DWG Analyzer",
    page_icon="📐",
    layout="wide"
)

st.title("Real DWG File Analyzer")
st.write("This tool actually parses DWG files and shows real data - no mock results")

# Debug info
st.sidebar.header("Debug Info")
st.sidebar.write(f"Max upload size: 200MB")
st.sidebar.write(f"Supported formats: DWG, DXF")

# File uploader with better error handling
uploaded_file = st.file_uploader(
    "Upload DWG/DXF file", 
    type=['dwg', 'dxf'],
    help="Select a DWG or DXF file (max 200MB)",
    key="dwg_uploader"
)

if uploaded_file and uploaded_file.size > 0:
    st.sidebar.write(f"File: {uploaded_file.name}")
    st.sidebar.write(f"Size: {uploaded_file.size / 1024:.1f} KB")
    # Validate file
    if uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
        st.error("File too large. Please use a file under 200MB.")
        st.stop()
    
    # Create temporary file with correct extension
    file_ext = uploaded_file.name.lower().split('.')[-1]
    
    with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp:
        try:
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) == 0:
                st.error("File appears to be empty")
                st.stop()
            tmp.write(file_bytes)
            tmp_path = tmp.name
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    
    try:
        doc = ezdxf.readfile(tmp_path)
        modelspace = doc.modelspace()
        
        st.success("File parsed successfully!")
        
        # Get real entities
        polylines = list(modelspace.query('LWPOLYLINE'))
        lines = list(modelspace.query('LINE'))
        circles = list(modelspace.query('CIRCLE'))
        
        st.write(f"Found {len(polylines)} polylines, {len(lines)} lines, {len(circles)} circles")
        
        # Show layers
        layers = [layer.dxf.name for layer in doc.layers]
        st.write("Layers:", layers)
        
        # Real room detection
        rooms = []
        for poly in polylines:
            if poly.closed and len(list(poly.get_points())) >= 3:
                points = [(p[0], p[1]) for p in poly.get_points()]
                area = Polygon(points).area
                if area > 1:  # Filter small areas
                    rooms.append({
                        'points': points,
                        'area': area,
                        'layer': poly.dxf.layer
                    })
        
        if rooms:
            st.write(f"Found {len(rooms)} rooms/zones")
            
            # Visualization
            fig = go.Figure()
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, room in enumerate(rooms):
                points = room['points']
                x = [p[0] for p in points] + [points[0][0]]
                y = [p[1] for p in points] + [points[0][1]]
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    fill='toself',
                    name=f"Room {i+1} ({room['area']:.1f} m²)",
                    fillcolor=colors[i % len(colors)]
                ))
            
            st.plotly_chart(fig)
            
            # Real data table
            for i, room in enumerate(rooms):
                st.write(f"Room {i+1}: {room['area']:.2f} m² on layer '{room['layer']}'")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        os.unlink(tmp_path)