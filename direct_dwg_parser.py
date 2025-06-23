import streamlit as st
import ezdxf
import os
import json
import plotly.graph_objects as go
from pathlib import Path
from shapely.geometry import Polygon

st.set_page_config(page_title="Direct DWG Parser", layout="wide")

def parse_dwg_file(file_path):
    """Parse DWG file and extract room data"""
    try:
        doc = ezdxf.readfile(file_path)
        msp = doc.modelspace()
        
        # Extract entities
        results = {
            'layers': [layer.dxf.name for layer in doc.layers],
            'entities': {},
            'rooms': []
        }
        
        # Count entities
        for entity_type in ['LWPOLYLINE', 'LINE', 'CIRCLE', 'ARC', 'POLYLINE']:
            entities = list(msp.query(entity_type))
            results['entities'][entity_type] = len(entities)
        
        # Extract rooms from closed polylines
        for poly in msp.query('LWPOLYLINE'):
            if poly.closed:
                try:
                    points = [(p[0], p[1]) for p in poly.get_points()]
                    if len(points) >= 3:
                        area = abs(Polygon(points).area)
                        if area > 1:  # Filter small areas
                            results['rooms'].append({
                                'points': points,
                                'area': area,
                                'layer': poly.dxf.layer,
                                'perimeter': Polygon(points).length
                            })
                except:
                    continue
        
        return results
    except Exception as e:
        return {'error': str(e)}

def create_room_visualization(rooms):
    """Create plotly visualization of rooms"""
    fig = go.Figure()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, room in enumerate(rooms):
        points = room['points']
        x = [p[0] for p in points] + [points[0][0]]
        y = [p[1] for p in points] + [points[0][1]]
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            fill='toself',
            fillcolor=colors[i % len(colors)],
            line=dict(color='black', width=1),
            name=f"Room {i+1} ({room['area']:.1f})",
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Room Layout from DWG",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        showlegend=True
    )
    fig.update_yaxis(scaleanchor="x", scaleratio=1)
    return fig

st.title("Direct DWG File Parser")
st.write("Place your DWG/DXF files in the 'sample_files' directory and select them below.")

# Create sample files directory
sample_dir = Path("sample_files")
sample_dir.mkdir(exist_ok=True)

# Check for existing files
dwg_files = list(sample_dir.glob("*.dwg")) + list(sample_dir.glob("*.dxf"))

if dwg_files:
    st.success(f"Found {len(dwg_files)} DWG/DXF files")
    
    # File selector
    selected_file = st.selectbox(
        "Select file to analyze:",
        options=[f.name for f in dwg_files]
    )
    
    if selected_file and st.button("Analyze Selected File"):
        file_path = sample_dir / selected_file
        
        with st.spinner("Parsing DWG file..."):
            results = parse_dwg_file(str(file_path))
        
        if 'error' in results:
            st.error(f"Error parsing file: {results['error']}")
        else:
            st.success("File parsed successfully!")
            
            # Display results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("File Information")
                st.write(f"**File:** {selected_file}")
                st.write(f"**Layers:** {len(results['layers'])}")
                
                st.subheader("Entities Found")
                for entity_type, count in results['entities'].items():
                    st.write(f"- {entity_type}: {count}")
                
                st.subheader("Rooms Detected")
                st.write(f"Total rooms: {len(results['rooms'])}")
                
                if results['rooms']:
                    for i, room in enumerate(results['rooms']):
                        st.write(f"Room {i+1}:")
                        st.write(f"  - Area: {room['area']:.2f}")
                        st.write(f"  - Layer: {room['layer']}")
            
            with col2:
                if results['rooms']:
                    st.subheader("Room Visualization")
                    fig = create_room_visualization(results['rooms'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No rooms detected in this file")
            
            # Export data
            if st.button("Export Analysis"):
                export_data = json.dumps(results, indent=2, default=str)
                st.download_button(
                    "Download JSON Report",
                    export_data,
                    f"{selected_file}_analysis.json",
                    "application/json"
                )

else:
    st.info("No DWG/DXF files found in the sample_files directory.")
    st.write("To use this parser:")
    st.write("1. Copy your DWG/DXF files to the 'sample_files' folder")
    st.write("2. Refresh this page")
    st.write("3. Select and analyze your files")

# Alternative: Text input for file path
st.subheader("Or specify file path directly")
file_path_input = st.text_input("Enter full path to DWG/DXF file:")

if file_path_input and st.button("Parse from Path"):
    if os.path.exists(file_path_input):
        results = parse_dwg_file(file_path_input)
        if 'error' not in results:
            st.success("File parsed successfully!")
            st.json(results)
        else:
            st.error(results['error'])
    else:
        st.error("File not found")

# Show current directory structure
with st.expander("Debug: Show Directory Structure"):
    st.write(f"Current directory: {os.getcwd()}")
    st.write(f"Sample directory: {sample_dir.absolute()}")
    st.write("Files in sample_files:")
    for f in sample_dir.iterdir():
        st.write(f"- {f.name} ({f.stat().st_size} bytes)")