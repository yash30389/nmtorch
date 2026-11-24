import cv2
import json
import numpy as np
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from app.server import DashboardServer, DashboardHandler
from app.vision import decode_movenet_pose, decode_object_detections


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
            page = response.read()
            self.assertIn(b"Visual monitor", page)
            self.assertNotIn(b"Network interfaces", page)

        with urllib.request.urlopen(self.base_url + "/health") as response:
            self.assertEqual(json.load(response), {"status": "ok"})

    def test_network_status_endpoint_is_removed(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.base_url + "/api/status")
        self.assertEqual(error.exception.code, 404)

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
        self.assertEqual(result["people_count"], 0)

    def test_movenet_pose_coordinates_map_from_crop_to_frame(self):
        prediction = np.zeros((1, 1, 1, 17, 3), dtype=np.float32)
        prediction[0, 0, 0, 0] = [0.25, 0.5, 0.8]
        pose = decode_movenet_pose(prediction, (10, 20, 110, 220), 320, 240)
        self.assertEqual(pose[0]["name"], "nose")
        self.assertAlmostEqual(pose[0]["x"], 0.1875)
        self.assertAlmostEqual(pose[0]["y"], 70 / 240)
        self.assertAlmostEqual(pose[0]["score"], 0.8)

    def test_movenet_pose_rejects_unexpected_landmark_count(self):
        with self.assertRaisesRegex(ValueError, "unsupported landmark count"):
            decode_movenet_pose(np.zeros((1, 5, 3)), (0, 0, 10, 10), 320, 240)

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
