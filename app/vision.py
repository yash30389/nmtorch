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
OBJECT_CONFIDENCE_THRESHOLD = 0.60
OBJECT_NMS_THRESHOLD = 0.45
MIN_OBJECT_SIDE_PIXELS = 6
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


def decode_object_detections(prediction, frame_width, frame_height):
    candidates_by_class = {}
    for item in np.asarray(prediction).reshape(-1, 7):
        if not np.isfinite(item[1]):
            continue
        confidence = float(item[2])
        class_index = int(item[1])
        if (
            not np.isfinite(confidence)
            or confidence < OBJECT_CONFIDENCE_THRESHOLD
            or class_index <= 0
            or class_index >= len(MODEL_CLASSES)
        ):
            continue

        coordinates = item[3:7]
        if not np.all(np.isfinite(coordinates)):
            continue
        left = max(0, min(frame_width, int(coordinates[0] * frame_width)))
        top = max(0, min(frame_height, int(coordinates[1] * frame_height)))
        right = max(left, min(frame_width, int(coordinates[2] * frame_width)))
        bottom = max(top, min(frame_height, int(coordinates[3] * frame_height)))
        width = right - left
        height = bottom - top
        if width < MIN_OBJECT_SIDE_PIXELS or height < MIN_OBJECT_SIDE_PIXELS:
            continue
        candidates_by_class.setdefault(class_index, []).append(
            (left, top, width, height, confidence)
        )

    detections = []
    for class_index, candidates in candidates_by_class.items():
        boxes = [candidate[:4] for candidate in candidates]
        confidences = [candidate[4] for candidate in candidates]
        selected = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            OBJECT_CONFIDENCE_THRESHOLD,
            OBJECT_NMS_THRESHOLD,
        )
        if selected is None:
            continue
        for selected_index in np.asarray(selected, dtype=np.intp).reshape(-1):
            left, top, width, height, _confidence = candidates[int(selected_index)]
            detections.append(
                (
                    confidence,
                    {
                        "label": MODEL_CLASSES[class_index],
                        "x": left / frame_width,
                        "y": top / frame_height,
                        "width": width / frame_width,
                        "height": height / frame_height,
                    },
                )
            )

    ordered = sorted(detections, key=lambda item: item[0], reverse=True)
    return [detection for _confidence, detection in ordered[:12]]


class VisionAnalyzer:
    def __init__(self):
        cv2.setNumThreads(1)
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
        detections = decode_object_detections(prediction, width, height)
        for detection in detections:
            label = detection["label"]
            seen_objects[label] = seen_objects.get(label, 0) + 1

        return {
            "objects": [
                {"label": label, "count": count} for label, count in sorted(seen_objects.items())
            ],
            "detections": detections,
            "notice": "Limited object labels are estimates; position change does not identify an action.",
        }
