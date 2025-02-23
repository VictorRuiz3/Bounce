import os
import logging
import psutil
import time
import socket
from contextlib import closing

logger = logging.getLogger(__name__)

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(2)  # Add timeout to avoid hanging
            result = sock.connect_ex(('0.0.0.0', port))
            logger.debug(f"Port {port} status check result: {result}")
            return result == 0
    except Exception as e:
        logger.error(f"Error checking port {port}: {e}")
        return True

def wait_for_port_release(port: int, timeout: int = 30) -> bool:
    """Wait for port to be released."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not is_port_in_use(port):
            logger.info(f"Port {port} has been released")
            return True
        logger.debug(f"Waiting for port {port} to be released...")
        time.sleep(1)
    logger.error(f"Timeout waiting for port {port} to be released")
    return False

def find_process_using_port(port: int) -> int:
    """Find the process ID using a port using multiple methods."""
    try:
        logger.info(f"Attempting to find process using port {port}")

        # Try netstat method first (more reliable)
        import subprocess
        try:
            netstat_cmd = f"netstat -tulpn 2>/dev/null | grep :{port}"
            output = subprocess.check_output(netstat_cmd, shell=True).decode()
            logger.debug(f"Netstat output: {output}")

            pid_cmd = f"netstat -tulpn 2>/dev/null | grep :{port} | awk '{{print $7}}' | cut -d'/' -f1"
            pid = subprocess.check_output(pid_cmd, shell=True).decode().strip()
            if pid and pid.isdigit():
                logger.info(f"Found process {pid} using netstat")
                return int(pid)
        except subprocess.CalledProcessError:
            logger.debug("Netstat method failed, trying lsof")

        # Try lsof as backup
        try:
            lsof_cmd = f"lsof -ti :{port}"
            pid = subprocess.check_output(lsof_cmd, shell=True).decode().strip()
            if pid and pid.isdigit():
                logger.info(f"Found process {pid} using lsof")
                return int(pid)
        except subprocess.CalledProcessError:
            logger.debug("Lsof method failed, trying psutil")

        # Use psutil as last resort
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections('tcp'):
                    if conn.laddr.port == port:
                        logger.info(f"Found process {proc.pid} ({proc.name()}) using psutil")
                        return proc.pid
            except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
                logger.debug(f"Error checking process {proc.pid}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error finding process: {str(e)}")

    logger.info(f"No process found using port {port}")
    return None

def kill_process_on_port(port: int, force: bool = False) -> bool:
    """Find and kill process running on specified port with enhanced reliability."""
    try:
        if not is_port_in_use(port):
            logger.info(f"Port {port} is not in use")
            return True

        logger.info(f"Attempting to kill process on port {port}")
        pid = find_process_using_port(port)

        if pid:
            logger.info(f"Found process {pid} using port {port}")
            try:
                process = psutil.Process(pid)
                logger.info(f"Process name: {process.name()}, status: {process.status()}")

                process.terminate()
                try:
                    process.wait(timeout=5)
                    logger.info(f"Process {pid} terminated successfully")
                except psutil.TimeoutExpired:
                    if force:
                        logger.warning(f"Force killing process {pid}")
                        process.kill()
                    else:
                        logger.error(f"Failed to terminate process {pid}")
                        return False

                if wait_for_port_release(port):
                    logger.info(f"Successfully killed process on port {port}")
                    return True
                else:
                    logger.error(f"Port {port} still in use after killing process")
                    return False

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.error(f"Error terminating process {pid}: {str(e)}")
                return False
        else:
            logger.info(f"No process found using port {port}")
            # If no process found but port is in use, wait briefly to see if it gets released
            return wait_for_port_release(port, timeout=5)

    except Exception as e:
        logger.error(f"Error cleaning up port {port}: {str(e)}")
        return False

def find_available_port(start_port: int, max_attempts: int = 5) -> int:
    """Find an available port starting from start_port."""
    current_port = start_port
    attempts = 0

    while attempts < max_attempts:
        if not is_port_in_use(current_port):
            logger.info(f"Found available port: {current_port}")
            return current_port
        current_port += 1
        attempts += 1

    logger.error(f"Could not find available port after {max_attempts} attempts")
    raise RuntimeError(f"No available ports found starting from {start_port}")

def ensure_port_available(port: int, max_attempts: int = 3) -> bool:
    """Ensure port is available with multiple cleanup attempts."""
    for attempt in range(max_attempts):
        try:
            if not is_port_in_use(port):
                logger.info(f"Port {port} is already available")
                return True

            logger.info(f"Cleanup attempt {attempt + 1}/{max_attempts} for port {port}")
            if kill_process_on_port(port, force=(attempt == max_attempts - 1)):
                return True

            time.sleep(2)  # Wait before next attempt

        except Exception as e:
            logger.error(f"Error in cleanup attempt {attempt + 1}: {str(e)}")

    logger.error(f"Failed to secure port {port} after {max_attempts} attempts")
    return False