import os
from pathlib import Path

import cv2
import numpy as np


MODEL_CLASSES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorcycle",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv",
)
JPEG_START_OF_FRAME_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF
}


def _read_jpeg_dimensions(frame_bytes):
    if len(frame_bytes) < 4 or frame_bytes[:2] != b"\xff\xd8":
        raise ValueError("The frame is not a valid JPEG image")

    position = 2
    while position < len(frame_bytes):
        if frame_bytes[position] != 0xFF:
            position += 1
            continue
        while position < len(frame_bytes) and frame_bytes[position] == 0xFF:
            position += 1
        if position >= len(frame_bytes):
            break
        marker = frame_bytes[position]
        position += 1
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(frame_bytes):
            break
        segment_length = int.from_bytes(frame_bytes[position:position + 2], "big")
        if segment_length < 2 or position + segment_length > len(frame_bytes):
            break
        if marker in JPEG_START_OF_FRAME_MARKERS:
            if segment_length < 7:
                break
            height = int.from_bytes(frame_bytes[position + 3:position + 5], "big")
            width = int.from_bytes(frame_bytes[position + 5:position + 7], "big")
            if width and height:
                return width, height
            break
        if marker == 0xDA:
            break
        position += segment_length
    raise ValueError("The frame is not a valid JPEG image")


class VisionAnalyzer:
    def __init__(self):
        cv2.setNumThreads(1)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(str(cascade_path))
        if self.face_detector.empty():
            raise RuntimeError("The bundled face-presence classifier could not be loaded")

        model_dir = Path(os.environ.get("NMTORCH_MODEL_DIR", "/opt/nmtorch"))
        prototxt = model_dir / "MobileNetSSD_deploy.prototxt"
        weights = model_dir / "mobilenet_iter_73000.caffemodel"
        if not prototxt.is_file() or not weights.is_file():
            raise RuntimeError("The bundled object-detection model is missing")
        self.object_detector = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))

    def analyze(self, frame_bytes):
        width, height = _read_jpeg_dimensions(frame_bytes)
        if width > 640 or height > 480 or width * height > 307200:
            raise ValueError("Camera frames must be no larger than 640 x 480")
        image_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("The frame is not a valid JPEG image")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(24, 24),
        )
        detections = []
        for x, y, face_width, face_height in faces[:10]:
            detections.append(self._box("face present", x, y, face_width, face_height, width, height))

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
        try:
            self.object_detector.setInput(blob)
            prediction = self.object_detector.forward()
        except cv2.error as error:
            raise RuntimeError("The bundled object-detection model could not process this frame") from error
        seen_objects = {}
        for item in prediction[0, 0]:
            confidence = float(item[2])
            class_index = int(item[1])
            if confidence < 0.55 or class_index <= 0 or class_index >= len(MODEL_CLASSES):
                continue
            label = MODEL_CLASSES[class_index]
            left = max(0, min(width, int(item[3] * width)))
            top = max(0, min(height, int(item[4] * height)))
            right = max(left, min(width, int(item[5] * width)))
            bottom = max(top, min(height, int(item[6] * height)))
            if right == left or bottom == top:
                continue
            seen_objects[label] = seen_objects.get(label, 0) + 1
            detections.append(
                self._box(label, left, top, right - left, bottom - top, width, height, confidence)
            )

        return {
            "faces_present": len(faces),
            "objects": [
                {"label": label, "count": count} for label, count in sorted(seen_objects.items())
            ],
            "detections": detections[:20],
            "notice": "Labels are estimates, not identity or intent.",
        }

    @staticmethod
    def _box(label, x, y, box_width, box_height, frame_width, frame_height, confidence=None):
        result = {
            "label": label,
            "x": x / frame_width,
            "y": y / frame_height,
            "width": box_width / frame_width,
            "height": box_height / frame_height,
        }
        if confidence is not None:
            result["confidence"] = round(confidence, 2)
        return result
