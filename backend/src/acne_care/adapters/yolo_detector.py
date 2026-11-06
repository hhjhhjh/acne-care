from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from threading import Lock

from acne_care.application.analysis import ModelInfo
from acne_care.domain import BoundingBox, Detection


class YOLOAcneDetector:
    def __init__(self, weights: Path, *, model_id: str, expected_sha256: str | None,
                 confidence_threshold: float = 0.3, image_size: int = 640, device: str = "cpu", nms_iou: float = 0.7):
        if not weights.is_file():
            raise FileNotFoundError(f"model weights not found: {weights}")
        digest = sha256(weights.read_bytes()).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError("model weights SHA-256 does not match the configured model")
        from ultralytics import YOLO
        self._model = YOLO(str(weights))
        self._lock = Lock()
        self.model_id = model_id
        self.info = ModelInfo(model_id, digest, confidence_threshold, image_size, device, nms_iou)

    def detect(self, image: object) -> tuple[Detection, ...]:
        info = self.info
        with self._lock:
            result = self._model.predict(source=image, conf=info.confidence_threshold,
                                         imgsz=info.image_size, device=info.device, iou=info.nms_iou,
                                         verbose=False)[0]
            rows = result.boxes.data.cpu().tolist()
        return tuple(Detection(BoundingBox(*map(float, row[:4])), float(row[4]), int(row[5]))
                     for row in rows if int(row[5]) == 0)
