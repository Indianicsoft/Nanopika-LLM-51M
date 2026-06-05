import http.server
import socketserver
import os

PORT = 5501

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_GET(self):
        print(f"Request: {self.path}")
        return super().do_GET()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Serving files from: {os.getcwd()}")

with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"Server started at http://localhost:{PORT}")
    httpd.serve_forever()
