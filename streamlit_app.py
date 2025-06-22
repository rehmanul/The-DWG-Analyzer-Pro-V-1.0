
import subprocess
import sys

# Run the main app with correct Streamlit configuration
if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "5000",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableWebsocketCompression", "false"
    ])
