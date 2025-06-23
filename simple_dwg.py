import streamlit as st
import ezdxf
import tempfile
import os
import json
from pathlib import Path

# Minimal config
st.set_page_config(page_title="DWG Parser", layout="wide")

st.title("Simple DWG File Parser")

# Create upload directory
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# Method 1: Direct file upload
st.header("Method 1: File Upload")
uploaded_file = st.file_uploader("Upload DWG/DXF", type=['dwg', 'dxf'])

if uploaded_file:
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size} bytes")
    
    try:
        # Save to disk first
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Saved to: {file_path}")
        
        # Try to parse
        if st.button("Parse File"):
            try:
                doc = ezdxf.readfile(str(file_path))
                st.success("File parsed successfully!")
                
                # Show basic info
                layers = [layer.dxf.name for layer in doc.layers]
                st.write(f"Layers found: {layers}")
                
                # Count entities
                msp = doc.modelspace()
                entities = {}
                for entity_type in ['LINE', 'LWPOLYLINE', 'CIRCLE', 'ARC']:
                    count = len(list(msp.query(entity_type)))
                    entities[entity_type] = count
                
                st.json(entities)
                
            except Exception as e:
                st.error(f"Parse error: {str(e)}")
            
    except Exception as e:
        st.error(f"Upload error: {str(e)}")

# Method 2: Manual file selection
st.header("Method 2: Browse Files")
if st.button("List uploaded files"):
    files = list(upload_dir.glob("*"))
    if files:
        for f in files:
            st.write(f"- {f.name} ({f.stat().st_size} bytes)")
    else:
        st.write("No files found")

# Debug info
st.header("Debug Information")
st.write(f"Upload directory: {upload_dir.absolute()}")
st.write(f"Current working directory: {os.getcwd()}")
st.write("Streamlit version:", st.__version__)

try:
    st.write("ezdxf version:", ezdxf.__version__)
except:
    st.write("ezdxf: Not available")