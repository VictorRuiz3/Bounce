import subprocess
import sys
import time
import os

def launch_services():
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Start FastAPI backend
        backend_process = subprocess.Popen(
            [sys.executable, "backend/api/document_processor_service.py"],
            cwd=project_root
        )
        print("Started FastAPI backend service...")
        
        # Give the backend a moment to start
        time.sleep(2)
        
        # Start Streamlit frontend
        frontend_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "frontend/main.py"],
            cwd=project_root
        )
        print("Started Streamlit frontend...")
        
        # Keep the script running
        frontend_process.wait()
        
    except KeyboardInterrupt:
        print("\nShutting down services...")
        frontend_process.terminate()
        backend_process.terminate()
        sys.exit(0)
    except Exception as e:
        print(f"Error launching services: {e}")
        sys.exit(1)

if __name__ == "__main__":
    launch_services()
