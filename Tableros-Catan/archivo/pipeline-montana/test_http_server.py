import http.server
import socketserver
import threading
import urllib.request
import time
import os

print("=== VERIFYING WEBGL VIEWER LOCAL HTTP SERVING & BROWSER ACCESSIBILITY ===")

PORT = 8765
web_dir = r"Tableros Catan\Loseta_Montana_3D"

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=web_dir, **kwargs)
    def log_message(self, format, *args):
        pass # Quiet log

server = socketserver.TCPServer(("127.0.0.1", PORT), QuietHandler)
server_thread = threading.Thread(target=server.serve_forever)
server_thread.daemon = True
server_thread.start()
print(f"Local test HTTP server started on http://127.0.0.1:{PORT}")

try:
    url = f"http://127.0.0.1:{PORT}/catan_mountain_viewer.html"
    t_req = time.time()
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=5) as response:
        status = response.status
        content_type = response.headers.get('Content-Type')
        content_length = int(response.headers.get('Content-Length', 0))
        
        print(f"HTTP Response Status: {status} OK")
        print(f"Content-Type:         {content_type}")
        print(f"Content-Length:       {content_length:,} bytes ({content_length/1024/1024:.2f} MB)")
        print(f"Response Time:        {(time.time()-t_req)*1000:.2f} ms")
        
        assert status == 200, f"Unexpected HTTP status: {status}"
        assert content_length > 10000000, f"Content-Length too small: {content_length}"
        print(">>> LOCAL HTTP SERVER ACCESSIBILITY TEST: 100% SUCCESSFUL! <<<")
finally:
    server.shutdown()
    server.server_close()
    print("Local HTTP server closed cleanly.")
