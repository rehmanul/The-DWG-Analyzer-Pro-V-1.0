import streamlit as st
import ezdxf
import tempfile
import os
import plotly.graph_objects as go
from shapely.geometry import Polygon
import base64

st.set_page_config(
    page_title="Working DWG Analyzer",
    page_icon="📐",
    layout="wide"
)

def create_download_link(data, filename, text):
    """Create download link for data"""
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{text}</a>'
    return href

st.title("Working DWG File Analyzer")
st.write("Upload a DWG or DXF file to analyze its geometry")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Upload File")
    
    # Simple file uploader
    uploaded_file = st.file_uploader(
        "Select DWG/DXF file",
        type=['dwg', 'dxf'],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        st.write(f"File: {uploaded_file.name}")
        st.write(f"Size: {uploaded_file.size:,} bytes")
        
        if st.button("Analyze File", type="primary"):
            with st.spinner("Processing file..."):
                try:
                    # Create temp file
                    suffix = '.' + uploaded_file.name.split('.')[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name
                    
                    # Parse DWG
                    doc = ezdxf.readfile(temp_path)
                    msp = doc.modelspace()
                    
                    # Extract data
                    entities = {
                        'LWPOLYLINE': list(msp.query('LWPOLYLINE')),
                        'LINE': list(msp.query('LINE')),
                        'CIRCLE': list(msp.query('CIRCLE')),
                        'ARC': list(msp.query('ARC'))
                    }
                    
                    # Find rooms (closed polylines)
                    rooms = []
                    for poly in entities['LWPOLYLINE']:
                        if poly.closed:
                            try:
                                points = [(p[0], p[1]) for p in poly.get_points()]
                                if len(points) >= 3:
                                    area = abs(Polygon(points).area)
                                    if area > 0.1:  # Filter tiny areas
                                        rooms.append({
                                            'points': points,
                                            'area': area,
                                            'layer': poly.dxf.layer
                                        })
                            except:
                                continue
                    
                    # Store results
                    st.session_state.entities = entities
                    st.session_state.rooms = rooms
                    st.session_state.analysis_complete = True
                    
                    # Cleanup
                    os.unlink(temp_path)
                    
                    st.success(f"Analysis complete! Found {len(rooms)} rooms")
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    if 'temp_path' in locals():
                        try:
                            os.unlink(temp_path)
                        except:
                            pass

with col2:
    st.header("Results")
    
    if st.session_state.analysis_complete:
        entities = st.session_state.entities
        rooms = st.session_state.rooms
        
        # Summary
        st.subheader("File Summary")
        for entity_type, entity_list in entities.items():
            st.write(f"{entity_type}: {len(entity_list)} found")
        
        if rooms:
            st.subheader("Rooms Found")
            
            # Create visualization
            fig = go.Figure()
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'yellow', 'cyan']
            
            for i, room in enumerate(rooms):
                points = room['points']
                x_coords = [p[0] for p in points] + [points[0][0]]
                y_coords = [p[1] for p in points] + [points[0][1]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor=colors[i % len(colors)],
                    line=dict(color='black', width=1),
                    name=f"Room {i+1}",
                    opacity=0.7,
                    hovertemplate=f"Room {i+1}<br>Area: {room['area']:.2f} sq units<br>Layer: {room['layer']}<extra></extra>"
                ))
            
            fig.update_layout(
                title="Room Layout",
                xaxis_title="X",
                yaxis_title="Y",
                showlegend=True,
                height=500
            )
            fig.update_yaxis(scaleanchor="x", scaleratio=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Room details table
            st.subheader("Room Details")
            room_data = []
            for i, room in enumerate(rooms):
                room_data.append({
                    'Room': f"Room {i+1}",
                    'Area': f"{room['area']:.2f}",
                    'Layer': room['layer'],
                    'Points': len(room['points'])
                })
            
            st.table(room_data)
            
            # Export data
            st.subheader("Export")
            export_data = ""
            for i, room in enumerate(rooms):
                export_data += f"Room {i+1}:\n"
                export_data += f"  Area: {room['area']:.2f} sq units\n"
                export_data += f"  Layer: {room['layer']}\n"
                export_data += f"  Points: {len(room['points'])}\n\n"
            
            st.download_button(
                "Download Analysis Report",
                export_data,
                "room_analysis.txt",
                "text/plain"
            )
        else:
            st.info("No closed rooms found in the file")
    else:
        st.info("Upload and analyze a file to see results here")

# Footer
st.markdown("---")
st.write("This tool uses the ezdxf library to parse DWG/DXF files and extract geometric data.")