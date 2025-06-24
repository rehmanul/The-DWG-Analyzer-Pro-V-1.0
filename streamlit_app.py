
#!/usr/bin/env python3
"""
AI Architectural Space Analyzer PRO - Production Entry Point
Enterprise-grade architectural drawing analysis with AI-powered insights
"""

import subprocess
import sys
import os

def main():
    """Launch the production application with optimized configuration"""
    
    # Set production environment variables
    os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
    os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    os.environ.setdefault('STREAMLIT_SERVER_ENABLE_CORS', 'false')
    os.environ.setdefault('STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION', 'false')
    
    # Launch with production configuration
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "5000",
        "--server.address", "0.0.0.0"
    ]
    
    print("🚀 Starting AI Architectural Space Analyzer PRO...")
    print("📊 Production-ready DWG/DXF analysis platform")
    print("🌐 Server: http://0.0.0.0:5000")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
