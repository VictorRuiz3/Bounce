import subprocess
import sys
import time
import os

def launch_services():
    # Get the project root directory and add to PYTHONPATH
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = project_root
    
    try:
        # Start FastAPI backend
        backend_process = subprocess.Popen(
            [sys.executable, "backend/api/document_processor_service.py"],
            cwd=project_root,
            env={**os.environ, "PYTHONPATH": project_root}
        )
        print("Started FastAPI backend service on http://localhost:8002")
        
        # Give the backend a moment to start
        time.sleep(2)
        
        # Start Streamlit frontend with specific port and host
        frontend_process = subprocess.Popen(
            [
                sys.executable,
                "-m", "streamlit",
                "run", "frontend/main.py",
                "--server.port=5000",
                "--server.address=0.0.0.0",
                "--browser.serverAddress=localhost",
                "--server.headless=true"
            ],
            cwd=project_root,
            env={**os.environ, "PYTHONPATH": project_root}
        )
        print("Started Streamlit frontend on http://localhost:5000")
        
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
