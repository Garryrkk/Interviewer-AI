import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ✅ Commands
OLLAMA_CMD = ["ollama", "serve"]
BACKEND_CMD = [
    sys.executable, "-m", "uvicorn", "app.main:app",
    "--host", "0.0.0.0", "--port", "8000", "--reload"
]
NGROK_CMD = [str(BASE_DIR / "ngrok.exe"), "http", "8000"]  # local ngrok binary

processes = []

def run_process(name, cmd, cwd=None):
    print(f"[INFO] Starting {name}... {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(cmd, cwd=cwd)
        processes.append(proc)
        return proc
    except Exception as e:
        print(f"[ERROR] Failed to start {name}: {e}")
        sys.exit(1)

def main():
    try:
        # 1. Start Ollama
        run_process("Ollama", OLLAMA_CMD)
        time.sleep(5)  # give Ollama time to boot

        # 2. Start Backend with Uvicorn
        run_process("Backend", BACKEND_CMD, cwd=BASE_DIR)
        time.sleep(3)  # wait for FastAPI server to boot

        # 3. Start Ngrok
        run_process("Ngrok", NGROK_CMD, cwd=BASE_DIR)

        print("\n✅ All services started successfully!")
        print("Ollama  → http://localhost:11434")
        print("Backend → http://localhost:8000")
        print("Ngrok   → Check terminal for the public URL (Forwarding)")

        # Optionally open backend docs in browser
        webbrowser.open("http://localhost:8000/docs")

        # Keep script alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping all services...")
        for p in processes:
            p.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
