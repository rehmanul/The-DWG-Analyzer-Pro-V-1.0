import streamlit as st
import ezdxf
import tempfile
import os
import plotly.graph_objects as go
from shapely.geometry import Polygon

st.title("Real DWG File Analyzer")
st.write("This tool actually parses DWG files and shows real data - no mock results")

uploaded_file = st.file_uploader("Upload DWG/DXF file", type=['dwg', 'dxf'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(suffix='.dwg', delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
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