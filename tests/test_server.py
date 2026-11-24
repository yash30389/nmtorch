import json
import cv2
import numpy as np
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from app.server import DashboardServer, DashboardHandler, _read_neighbors, _read_traffic_counters
from app.vision import decode_object_detections


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

    def test_camera_endpoint_requires_same_origin(self):
        request = urllib.request.Request(
            self.base_url + "/api/camera/analyze",
            data=b"not an image",
            headers={"Content-Type": "image/jpeg", "Origin": "http://example.test"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_camera_endpoint_rejects_non_jpeg_content_type(self):
        request = urllib.request.Request(
            self.base_url + "/api/camera/analyze",
            data=b"not an image",
            headers={"Content-Type": "text/plain", "Origin": self.base_url},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 415)

    def test_camera_endpoint_rejects_oversized_frames(self):
        request = urllib.request.Request(
            self.base_url + "/api/camera/analyze",
            data=b"x" * (256 * 1024 + 1),
            headers={"Content-Type": "image/jpeg", "Origin": self.base_url},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 413)

    @patch(
        "app.server.analyze_camera_frame",
        return_value={
            "objects": [],
            "detections": [],
        },
    )
    def test_camera_endpoint_returns_current_estimates_without_timestamps(self, analyze):
        ok, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertTrue(ok)
        request = urllib.request.Request(
            self.base_url + "/api/camera/analyze",
            data=encoded.tobytes(),
            headers={"Content-Type": "image/jpeg", "Origin": self.base_url},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
        self.assertNotIn("timestamp", result)
        analyze.assert_called_once()

    def test_camera_stop_route_is_not_exposed(self):
        request = urllib.request.Request(
            self.base_url + "/api/camera/stop",
            data=b"",
            headers={"Origin": self.base_url},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 404)


class VisionAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.vision import VisionAnalyzer

        cls.analyzer = VisionAnalyzer()

    def test_blank_frame_has_no_detections_on_first_frame(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertTrue(ok)
        result = self.analyzer.analyze(encoded.tobytes())
        self.assertEqual(result["objects"], [])
        self.assertEqual(result["detections"], [])

    def test_object_decoder_filters_weak_boxes_and_suppresses_duplicates(self):
        prediction = np.array(
            [[[
                [0, 15, 0.92, 0.10, 0.10, 0.50, 0.50],
                [0, 15, 0.83, 0.11, 0.11, 0.49, 0.49],
                [0, 7, 0.59, 0.10, 0.10, 0.50, 0.50],
                [0, 99, 0.97, 0.10, 0.10, 0.50, 0.50],
                [0, 5, 0.97, 0.20, 0.20, 0.21, 0.21],
            ]]],
            dtype=np.float32,
        )
        detections = decode_object_detections(prediction, 320, 240)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "person")
        self.assertAlmostEqual(detections[0]["x"], 0.1)

    def test_object_decoder_keeps_overlapping_different_classes(self):
        prediction = np.array(
            [[[
                [0, 15, 0.92, 0.10, 0.10, 0.50, 0.50],
                [0, 7, 0.88, 0.10, 0.10, 0.50, 0.50],
            ]]],
            dtype=np.float32,
        )
        detections = decode_object_detections(prediction, 320, 240)
        self.assertEqual({item["label"] for item in detections}, {"person", "car"})

    def test_invalid_or_oversized_frames_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid JPEG"):
            self.analyzer.analyze(b"not a JPEG")

        ok, encoded = cv2.imencode(".jpg", np.zeros((481, 641, 3), dtype=np.uint8))
        self.assertTrue(ok)
        with self.assertRaisesRegex(ValueError, "640 x 480"):
            self.analyzer.analyze(encoded.tobytes())


if __name__ == "__main__":
    unittest.main()
