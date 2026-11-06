from __future__ import annotations

import io
import shutil
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from acne_care.domain import Detection
from acne_care.ports.acne_detector import AcneDetector


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    weights_sha256: str
    confidence_threshold: float
    image_size: int
    device: str
    nms_iou: float = 0.7


class AnalysisRepository(Protocol):
    def insert(self, record: dict[str, Any], detections: tuple[Detection, ...]) -> None: ...
    def get(self, analysis_id: str) -> dict[str, Any] | None: ...
    def list(self, limit: int, offset: int) -> dict[str, Any]: ...


class InvalidImage(ValueError):
    pass


def decode_image(data: bytes, max_pixels: int) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImage("JPG, PNG, WEBP 사진을 선택해 주세요.")
                if source.width * source.height > max_pixels or source.width < 16 or source.height < 16:
                    raise InvalidImage("사진 크기는 가로·세로 16픽셀 이상, 전체 2천만 픽셀 이하로 선택해 주세요.")
                if getattr(source, "n_frames", 1) > 1:
                    raise InvalidImage("움직이는 이미지 대신 사진 한 장을 선택해 주세요.")
                source.verify()
            with Image.open(io.BytesIO(data)) as source:
                return ImageOps.exif_transpose(source).convert("RGB")
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise InvalidImage("사진을 읽을 수 없습니다. 정상적인 JPG, PNG, WEBP 파일을 선택해 주세요.") from error


class AnalysisService:
    def __init__(self, detector: AcneDetector, model_info: ModelInfo,
                 repository: AnalysisRepository, storage_root: Path, *, max_pixels: int = 20_000_000):
        if detector.model_id != model_info.model_id:
            raise ValueError("detector and model metadata must agree")
        self.detector = detector
        self.model_info = model_info
        self.repository = repository
        self.storage_root = storage_root
        self.max_pixels = max_pixels

    def analyze(self, data: bytes, captured_on: date) -> dict[str, Any]:
        if captured_on > date.today():
            raise ValueError("촬영일은 오늘 또는 이전 날짜를 선택해 주세요.")
        image = decode_image(data, self.max_pixels)
        started = time.perf_counter()
        detections = tuple(self.detector.detect(image))
        inference_seconds = time.perf_counter() - started
        for detection in detections:
            detection.box.validate_image_bounds(width=image.width, height=image.height)
            if detection.class_id != 0 or detection.confidence < self.model_info.confidence_threshold:
                raise ValueError("detector returned a class or confidence outside the configured contract")
        analysis_id = str(uuid4())
        directory = self.storage_root / analysis_id
        directory.mkdir(parents=True, exist_ok=False)
        try:
            # Saving a fresh RGB JPEG drops source EXIF and location metadata.
            image.save(directory / "original.jpg", quality=95)
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            line_width = max(2, round(image.width / 350))
            font = ImageFont.load_default(size=max(14, round(image.width / 45)))
            for index, detection in enumerate(detections):
                box = detection.box
                draw.rectangle((box.x_min, box.y_min, box.x_max, box.y_max), outline="#37dcc0", width=line_width)
                text = f"{index+1} {detection.confidence:.0%}"
                x, y = box.x_min, max(0, box.y_min - max(19, round(image.width / 35)))
                bounds = draw.textbbox((x, y), text, font=font)
                draw.rectangle(bounds, fill="#123c36")
                draw.text((x, y), text, fill="white", font=font)
            annotated.save(directory / "annotated.jpg", quality=95)
            info = self.model_info
            record = {
                "id": analysis_id, "captured_on": captured_on.isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "width": image.width, "height": image.height,
                "model_id": info.model_id, "weights_sha256": info.weights_sha256,
                "confidence_threshold": info.confidence_threshold, "image_size": info.image_size,
                "device": info.device, "nms_iou": info.nms_iou, "inference_seconds": inference_seconds,
                "original_path": f"{analysis_id}/original.jpg", "annotated_path": f"{analysis_id}/annotated.jpg",
            }
            self.repository.insert(record, detections)
        except Exception:
            shutil.rmtree(directory)
            raise
        result = self.repository.get(analysis_id)
        if result is None:
            raise RuntimeError("analysis was not persisted")
        return result
