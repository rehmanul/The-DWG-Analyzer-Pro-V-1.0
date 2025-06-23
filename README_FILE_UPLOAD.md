# DWG File Upload Instructions

The Streamlit file uploader has persistent 400 errors. Here's the workaround:

## Method 1: Direct File Copy
1. Copy your DWG/DXF files to the `sample_files` folder in this project
2. Use the dropdown in the app to select your file
3. Click "Analyze Selected File"

## Method 2: File Path Input  
1. Enter the full path to your DWG file in the text input
2. Click "Parse from Path"

## Example Files Supported
- .dwg files (AutoCAD native format)
- .dxf files (Drawing Exchange Format)

The parser will extract:
- Room boundaries (closed polylines)
- Area calculations
- Layer information
- Entity counts
- Interactive visualization

This approach completely avoids the upload system that's causing errors.