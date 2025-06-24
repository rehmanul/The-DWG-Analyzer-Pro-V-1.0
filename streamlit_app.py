
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
    os.environ.setdefault('STREAMLIT_SERVER_ENABLE_CORS', 'true')
    os.environ.setdefault('STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION', 'false')
    os.environ.setdefault('STREAMLIT_SERVER_MAX_UPLOAD_SIZE', '200')
    os.environ.setdefault('STREAMLIT_SERVER_MAX_MESSAGE_SIZE', '200')
    
    # Launch with production configuration
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "5000",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "true",
        "--server.enableXsrfProtection", "false",
        "--server.maxUploadSize", "200",
        "--server.maxMessageSize", "200",
        "--server.fileWatcherType", "none",
        "--server.allowRunOnSave", "false"
    ]
    
    print("🚀 Starting AI Architectural Space Analyzer PRO...")
    print("📊 Production-ready DWG/DXF analysis platform")
    print("🌐 Server: http://0.0.0.0:5000")
    print("📁 Max file size: 200MB")
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
