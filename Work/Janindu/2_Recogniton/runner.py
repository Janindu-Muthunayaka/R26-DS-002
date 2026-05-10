import os
import subprocess
import threading
import time
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Configuration
PORT = 8000
VENV_PYTHON = os.path.join(os.getcwd(), "venv311", "Scripts", "python.exe")
MAIN_SCRIPT = "MainRecognize.py"

# Global state
PIPELINE_LOCK = threading.Lock()
CURRENT_PROCESS = None
LOGS = []
IS_DONE = False
RETURN_CODE = 0

def pipeline_worker(command):
    global CURRENT_PROCESS, IS_DONE, RETURN_CODE
    
    CURRENT_PROCESS = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace', # Prevent thread crash on weird characters
    )

    def log_reader(proc):
        global IS_DONE
        try:
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                LOGS.append(line.strip())
        except Exception as e:
            LOGS.append(f"  [RUNNER ERROR] Logging thread failed: {e}")
        finally:
            proc.stdout.close()

    threading.Thread(target=log_reader, args=(CURRENT_PROCESS,), daemon=True).start()
    
    # Wait for process to actually finish
    CURRENT_PROCESS.wait()
    RETURN_CODE = CURRENT_PROCESS.returncode
    IS_DONE = True

class PipelineRunnerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global CURRENT_PROCESS, LOGS, IS_DONE, RETURN_CODE
        
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('Main.html', 'rb') as f:
                self.wfile.write(f.read())
        
        elif self.path.startswith('/run'):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            sample = params.get('sample', ['10'])[0]

            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            with PIPELINE_LOCK:
                # Start new pipeline only if none is running
                if CURRENT_PROCESS is None or IS_DONE:
                    LOGS.clear()
                    IS_DONE = False
                    RETURN_CODE = 0
                    
                    command = [VENV_PYTHON, MAIN_SCRIPT]
                    if sample == 'all':
                        command.append('--fullset')
                    else:
                        command.extend(['--sample', sample])
                        
                    self.log_message(f"Starting pipeline: {' '.join(command)}")
                    thread = threading.Thread(target=pipeline_worker, args=(command,))
                    thread.daemon = True
                    thread.start()

            # Stream logs to client
            log_idx = 0
            try:
                while True:
                    if log_idx < len(LOGS):
                        msg = json.dumps({"type": "log", "content": LOGS[log_idx]})
                        self.wfile.write(f"data: {msg}\n\n".encode('utf-8'))
                        self.wfile.flush()
                        log_idx += 1
                    elif IS_DONE:
                        # Flush any remaining logs that might have come in
                        while log_idx < len(LOGS):
                            msg = json.dumps({"type": "log", "content": LOGS[log_idx]})
                            self.wfile.write(f"data: {msg}\n\n".encode('utf-8'))
                            self.wfile.flush()
                            log_idx += 1
                            
                        final_msg = json.dumps({"type": "done", "status": RETURN_CODE})
                        self.wfile.write(f"data: {final_msg}\n\n".encode('utf-8'))
                        self.wfile.flush()
                        break
                    else:
                        time.sleep(0.5)
            except Exception as e:
                # Client disconnected (e.g. browser timeout)
                # We do NOT terminate the process! It continues in the background thread.
                pass

        else:
            # Serve static files (results, images, etc.)
            filepath = self.path.lstrip('/')
            if os.path.exists(filepath):
                self.send_response(200)
                # Basic mime-type handling
                if filepath.endswith('.html'): self.send_header('Content-type', 'text/html')
                elif filepath.endswith('.png'): self.send_header('Content-type', 'image/png')
                elif filepath.endswith('.css'): self.send_header('Content-type', 'text/css')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)

def run_server():
    server = ThreadingHTTPServer(('localhost', PORT), PipelineRunnerHandler)
    print(f"--- OCR Pipeline Runner Started ---")
    print(f"Open your browser at: http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop.")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
