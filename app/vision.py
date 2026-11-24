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
MAX_POSE_DETECTIONS = 5
POSE_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
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


def decode_movenet_pose(prediction, crop_rect, frame_width, frame_height):
    landmarks = np.asarray(prediction, dtype=np.float32).reshape(-1, 3)
    if len(landmarks) != len(POSE_KEYPOINT_NAMES):
        raise ValueError("The pose model returned an unsupported landmark count")
    crop_left, crop_top, crop_right, crop_bottom = crop_rect
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("The person crop is empty")

    keypoints = []
    for name, (y, x, score) in zip(POSE_KEYPOINT_NAMES, landmarks):
        if not np.all(np.isfinite((x, y, score))):
            continue
        keypoints.append(
            {
                "name": name,
                "x": max(0.0, min(1.0, (crop_left + float(x) * crop_width) / frame_width)),
                "y": max(0.0, min(1.0, (crop_top + float(y) * crop_height) / frame_height)),
                "score": max(0.0, min(1.0, float(score))),
            }
        )
    return keypoints


class VisionAnalyzer:
    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter

        cv2.setNumThreads(1)
        model_dir = Path(os.environ.get("NMTORCH_MODEL_DIR", "/opt/nmtorch"))
        prototxt = model_dir / "MobileNetSSD_deploy.prototxt"
        weights = model_dir / "mobilenet_iter_73000.caffemodel"
        pose_model = model_dir / "movenet_lightning_float16.tflite"
        if not prototxt.is_file() or not weights.is_file():
            raise RuntimeError("The bundled object-detection model is missing")
        if not pose_model.is_file():
            raise RuntimeError("The bundled human-pose model is missing")
        self.object_detector = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
        self.pose_interpreter = Interpreter(model_path=str(pose_model), num_threads=2)
        self.pose_interpreter.allocate_tensors()
        self.pose_input = self.pose_interpreter.get_input_details()[0]
        self.pose_output = self.pose_interpreter.get_output_details()[0]
        self.pose_input_height = int(self.pose_input["shape"][1])
        self.pose_input_width = int(self.pose_input["shape"][2])

    def _pose_for_person(self, frame, detection):
        frame_height, frame_width = frame.shape[:2]
        left = int(detection["x"] * frame_width)
        top = int(detection["y"] * frame_height)
        right = left + int(detection["width"] * frame_width)
        bottom = top + int(detection["height"] * frame_height)
        box_width = max(right - left, bottom - top)
        side = max(1, int(box_width * 1.25))
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        crop_left = center_x - side // 2
        crop_top = center_y - side // 2
        crop_right = crop_left + side
        crop_bottom = crop_top + side

        bounded_left = max(0, crop_left)
        bounded_top = max(0, crop_top)
        bounded_right = min(frame_width, crop_right)
        bounded_bottom = min(frame_height, crop_bottom)
        crop = frame[bounded_top:bounded_bottom, bounded_left:bounded_right]
        if crop.size == 0:
            return []
        padded = cv2.copyMakeBorder(
            crop,
            max(0, -crop_top),
            max(0, crop_bottom - frame_height),
            max(0, -crop_left),
            max(0, crop_right - frame_width),
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        resized = cv2.resize(padded, (self.pose_input_width, self.pose_input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb.astype(self.pose_input["dtype"]), axis=0)
        self.pose_interpreter.set_tensor(self.pose_input["index"], tensor)
        try:
            self.pose_interpreter.invoke()
        except RuntimeError as error:
            raise RuntimeError("The bundled pose model could not process this frame") from error
        prediction = self.pose_interpreter.get_tensor(self.pose_output["index"])
        return decode_movenet_pose(
            prediction,
            (crop_left, crop_top, crop_right, crop_bottom),
            frame_width,
            frame_height,
        )

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
        pose_count = 0
        for detection in detections:
            label = detection["label"]
            seen_objects[label] = seen_objects.get(label, 0) + 1
            if label == "person" and pose_count < MAX_POSE_DETECTIONS:
                detection["pose"] = self._pose_for_person(frame, detection)
                pose_count += 1

        return {
            "objects": [
                {"label": label, "count": count} for label, count in sorted(seen_objects.items())
            ],
            "detections": detections,
            "people_count": seen_objects.get("person", 0),
            "notice": "Object labels and pose-based activity estimates may be inaccurate.",
        }
