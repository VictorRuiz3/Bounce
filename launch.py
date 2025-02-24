
import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def start_fastapi():
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([
        sys.executable,
        "backend/api/document_processor_service.py"
    ])

def start_streamlit():
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([
        sys.executable,
        "-m", "streamlit",
        "run",
        "frontend/main.py",
        "--server.port=5000",
        "--server.address=0.0.0.0"
    ])

def main():
    print("Starting RAG Document Processing System...")
    
    # Start FastAPI in a separate thread
    fastapi_thread = Thread(target=start_fastapi)
    fastapi_thread.daemon = True
    fastapi_thread.start()
    
    print("Backend server starting on port 8002...")
    time.sleep(2)  # Give FastAPI time to start
    
    print("Starting frontend on port 5000...")
    start_streamlit()

if __name__ == "__main__":
    main()
