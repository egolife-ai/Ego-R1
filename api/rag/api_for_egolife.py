import os
import signal
import time
import subprocess
from typing import Dict, Any
from pathlib import Path

import yaml
import psutil


class APIManager:
    def __init__(self, config_path: str = "configs/egolife.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.processes: Dict[str, subprocess.Popen] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _kill_process_on_port(self, port: int) -> None:
        """Kill any process running on the specified port."""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        os.kill(proc.pid, signal.SIGKILL)
                        print(f"Killed process on port {port}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def start_api(self, api_name: str, api_config: Dict[str, Any]) -> None:
        """Start a single API instance."""
        identity = api_config['identity']
        port = api_config['port']
        data_dir = Path(self.config['base']['data_dir'])
        

        # Kill any existing process on the port
        self._kill_process_on_port(port)

        # Prepare log file paths
        day_log = log_dir /f"{api_name}_{identity}"/ f"{api_name}_{identity}_1day.json"
        hour_log = log_dir /f"{api_name}_{identity}"/ f"{api_name}_{identity}_1hour.json"
        min_log = log_dir / f"{api_name}_{identity}"/f"{api_name}_{identity}_10min.json"

        # Start the API
        print(f"Starting {api_name} ({identity}) on port {port}")
        cmd = [
            "python3",
            f"api_{api_name}.py",
            "--day_log", str(day_log),
            "--hour_log", str(hour_log),
            "--min_log", str(min_log),
            "--port", str(port)
        ]
        os.makedirs(f"logs", exist_ok=True)
        # Start process and redirect output to log file
        with open(f"logs/{api_name}_server.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            self.processes[api_name] = process

        # Wait for server to start
        time.sleep(2)

        # Check if process is still running
        if process.poll() is None:
            print(f"✓ {api_name} started successfully")
        else:
            print(f"✗ Failed to start {api_name}")

    def start_all(self) -> None:
        """Start all APIs defined in the configuration."""
        for api_name, api_config in self.config['apis'].items():
            self.start_api(api_name, api_config)
        print("\nAll APIs have been started. Check individual log files for details.")

    def stop_all(self) -> None:
        """Stop all running API processes."""
        for api_name, process in self.processes.items():
            if process.poll() is None:
                process.terminate()
                print(f"Stopped {api_name}")
        self.processes.clear()


def main():
    manager = APIManager()
    try:
        manager.start_all()
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all APIs...")
        manager.stop_all()


if __name__ == "__main__":
    main()