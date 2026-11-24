import json
import os
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


APP_DIR = Path(__file__).resolve().parent
STATIC_FILES = {
    "/static/app.css": ("app.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
POLL_INTERVAL_SECONDS = 4
MAX_CAMERA_FRAME_BYTES = 256 * 1024
CAMERA_FRAME_CONTENT_TYPE = "image/jpeg"
_camera_analyzer = None
_camera_lock = threading.Lock()


def _read_optional(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _read_traffic_counters():
    counters = {}
    with open("/proc/net/dev", encoding="utf-8") as device_file:
        for line in device_file.readlines()[2:]:
            if ":" not in line:
                continue
            name, values = line.split(":", 1)
            numbers = values.split()
            if len(numbers) >= 9:
                counters[name.strip()] = {
                    "received_bytes": int(numbers[0]),
                    "transmitted_bytes": int(numbers[8]),
                }
    return counters


def _read_ipv4_address(interface_name):
    try:
        import fcntl
    except ImportError:
        return None

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            request = struct.pack("256s", interface_name.encode("utf-8")[:15])
            result = fcntl.ioctl(sock.fileno(), 0x8915, request)
            return socket.inet_ntoa(result[20:24])
    except OSError:
        return None


def _read_interfaces():
    traffic = _read_traffic_counters()
    interfaces = []
    for name in sorted(os.listdir("/sys/class/net")):
        interface_dir = Path("/sys/class/net") / name
        counters = traffic.get(name, {})
        interfaces.append(
            {
                "name": name,
                "type": "wireless"
                if (interface_dir / "wireless").is_dir()
                else "container network",
                "state": _read_optional(interface_dir / "operstate") or "unknown",
                "ipv4": _read_ipv4_address(name),
                "mac": _read_optional(interface_dir / "address"),
                "received_bytes": counters.get("received_bytes", 0),
                "transmitted_bytes": counters.get("transmitted_bytes", 0),
            }
        )
    return interfaces


def _read_neighbors():
    neighbors = []
    with open("/proc/net/arp", encoding="utf-8") as arp_file:
        for line in arp_file.readlines()[1:]:
            fields = line.split()
            if len(fields) < 6:
                continue
            flags = int(fields[2], 16)
            neighbors.append(
                {
                    "address": fields[0],
                    "mac": fields[3],
                    "state": "reachable" if flags & 0x2 else "cached",
                    "interface": fields[5],
                }
            )
    return neighbors


def get_status():
    return {
        "scope": "Docker container only",
        "poll_interval_seconds": POLL_INTERVAL_SECONDS,
        "interfaces": _read_interfaces(),
        "neighbors": _read_neighbors(),
        "radios": {
            "wifi": {
                "available": False,
                "message": "Host Wi-Fi radios are not exposed to this container.",
            },
            "bluetooth": {
                "available": False,
                "message": "Host Bluetooth radios are not exposed to this container.",
            },
        },
    }


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
        if route == "/api/status":
            self._send_json(get_status())
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
