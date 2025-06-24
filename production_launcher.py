#!/usr/bin/env python3
"""
Production launcher for AI Architectural Space Analyzer PRO
Handles environment setup, database initialization, and application startup
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Set up production environment variables"""
    env_vars = {
        'STREAMLIT_SERVER_HEADLESS': 'true',
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false',
        'STREAMLIT_SERVER_ENABLE_CORS': 'false',
        'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION': 'false',
        'STREAMLIT_SERVER_MAX_UPLOAD_SIZE': '200',
        'STREAMLIT_SERVER_MAX_MESSAGE_SIZE': '200'
    }
    
    for key, value in env_vars.items():
        os.environ.setdefault(key, value)
    
    logger.info("Environment variables configured for production")

def check_dependencies():
    """Verify all required dependencies are installed"""
    required_packages = [
        'streamlit', 'ezdxf', 'shapely', 'plotly', 'pandas', 
        'numpy', 'matplotlib', 'psycopg2', 'sqlalchemy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        return False
    
    logger.info("All dependencies verified")
    return True

def initialize_database():
    """Initialize database connection and create tables if needed"""
    try:
        from src.database import DatabaseManager
        db_manager = DatabaseManager()
        db_manager.create_tables()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        logger.info("Application will use SQLite fallback")
        return False

def launch_application():
    """Launch the Streamlit application"""
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "5000",
        "--server.address", "0.0.0.0"
    ]
    
    logger.info("Starting AI Architectural Space Analyzer PRO")
    logger.info("Server: http://0.0.0.0:5000")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")

def main():
    """Main production launcher"""
    print("=" * 60)
    print("AI ARCHITECTURAL SPACE ANALYZER PRO")
    print("Enterprise-grade DWG/DXF Analysis Platform")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    # Initialize database
    initialize_database()
    
    # Launch application
    launch_application()

if __name__ == "__main__":
    main()