import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


APP_DIR = Path(__file__).resolve().parent
STATIC_FILES = {
    "/static/app.css": ("app.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_CAMERA_FRAME_BYTES = 256 * 1024
CAMERA_FRAME_CONTENT_TYPE = "image/jpeg"
_camera_analyzer = None
_camera_lock = threading.Lock()


def analyze_camera_frame(frame):
    global _camera_analyzer
    with _camera_lock:
        if _camera_analyzer is None:
            from app.vision import VisionAnalyzer

            _camera_analyzer = VisionAnalyzer()
        return _camera_analyzer.analyze(frame)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "NMTorch"
    sys_version = ""

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/health":
            self._send_json({"status": "ok"})
            return
        if route == "/":
            self._send_file(APP_DIR / "static" / "index.html", "text/html; charset=utf-8")
            return
        if route in STATIC_FILES:
            filename, content_type = STATIC_FILES[route]
            self._send_file(APP_DIR / "static" / filename, content_type)
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        route = urlsplit(self.path).path
        if route != "/api/camera/analyze":
            self._send_json({"error": "Not found"}, status=404)
            return
        if not self._is_same_origin_request():
            self._send_json({"error": "Same-origin requests only"}, status=403)
            return
        if self.headers.get_content_type() != CAMERA_FRAME_CONTENT_TYPE:
            self._send_json({"error": "Expected an image/jpeg frame"}, status=415)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, status=400)
            return
        if content_length <= 0 or content_length > MAX_CAMERA_FRAME_BYTES:
            self._send_json({"error": "Frame must be between 1 byte and 256 KiB"}, status=413)
            return
        frame_bytes = self.rfile.read(content_length)
        if len(frame_bytes) != content_length:
            self._send_json({"error": "Incomplete camera frame"}, status=400)
            return
        try:
            result = analyze_camera_frame(frame_bytes)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=422)
            return
        except RuntimeError as error:
            self._send_json({"error": str(error)}, status=503)
            return
        self._send_json(result)

    def _is_same_origin_request(self):
        origin = self.headers.get("Origin")
        if not origin:
            return False
        parsed_origin = urlsplit(origin)
        return parsed_origin.scheme in ("http", "https") and parsed_origin.netloc == self.headers.get("Host")

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _send_file(self, path, content_type):
        body = path.read_bytes()
        self._send(body, content_type)

    def _send(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    server = DashboardServer((host, port), DashboardHandler)
    print("NMTorch listening on {}:{}".format(host, port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
