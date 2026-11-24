import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from app.server import DashboardServer, DashboardHandler, _read_neighbors, _read_traffic_counters


class NetworkParsingTests(unittest.TestCase):
    @patch(
        "builtins.open",
        return_value=__import__("io").StringIO(
            "IP address       HW type     Flags       HW address            Mask     Device\n"
            "172.20.0.1       0x1         0x2         02:42:aa:bb:cc:dd     *        eth0\n"
            "172.20.0.2       0x1         0x0         00:00:00:00:00:00     *        eth0\n"
        ),
    )
    def test_neighbor_cache_distinguishes_valid_and_cached_entries(self, _open):
        self.assertEqual(
            _read_neighbors(),
            [
                {
                    "address": "172.20.0.1",
                    "mac": "02:42:aa:bb:cc:dd",
                    "state": "reachable",
                    "interface": "eth0",
                },
                {
                    "address": "172.20.0.2",
                    "mac": "00:00:00:00:00:00",
                    "state": "cached",
                    "interface": "eth0",
                },
            ],
        )

    @patch(
        "builtins.open",
        return_value=__import__("io").StringIO(
            "Inter-| Receive | Transmit\n"
            " face |bytes packets errs drop fifo frame compressed multicast|bytes packets errs drop fifo colls carrier compressed\n"
            "  eth0: 1024 8 0 0 0 0 0 1 2048 9 0 0 0 0 0 0\n"
        ),
    )
    def test_traffic_counters_read_container_bytes(self, _open):
        self.assertEqual(
            _read_traffic_counters(),
            {
                "eth0": {
                    "received_bytes": 1024,
                    "transmitted_bytes": 2048,
                }
            },
        )


class HttpEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = DashboardServer(("127.0.0.1", 0), DashboardHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = "http://127.0.0.1:{}".format(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_dashboard_and_health_endpoints(self):
        with urllib.request.urlopen(self.base_url + "/") as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"NMTorch", response.read())

        with urllib.request.urlopen(self.base_url + "/health") as response:
            self.assertEqual(json.load(response), {"status": "ok"})

    @patch(
        "app.server.get_status",
        return_value={
            "scope": "Docker container only",
            "interfaces": [],
            "neighbors": [],
            "radios": {
                "wifi": {"available": False},
                "bluetooth": {"available": False},
            },
        },
    )
    def test_status_endpoint_reports_scope_and_radio_boundaries(self, _status):
        with urllib.request.urlopen(self.base_url + "/api/status") as response:
            payload = json.load(response)
        self.assertEqual(payload["scope"], "Docker container only")
        self.assertFalse(payload["radios"]["wifi"]["available"])
        self.assertFalse(payload["radios"]["bluetooth"]["available"])

    def test_unknown_routes_return_404(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.base_url + "/not-found")
        self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
