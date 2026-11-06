from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_CONFIG_PATH = PROJECT_ROOT / "models/detector-baseline.yaml"
MODEL_CONFIG = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
if MODEL_CONFIG.get("framework") != "ultralytics":
    raise ValueError("the web service requires an ultralytics detector manifest")


@dataclass(frozen=True)
class Settings:
    storage_dir: Path = PROJECT_ROOT / "uploads/service"
    database_path: Path | None = None
    weights_path: Path = PROJECT_ROOT / MODEL_CONFIG["weights_path"]
    model_id: str = MODEL_CONFIG["model_id"]
    expected_sha256: str = MODEL_CONFIG["weights_sha256"]
    confidence_threshold: float = MODEL_CONFIG["confidence_threshold"]
    image_size: int = MODEL_CONFIG["input_size"]
    nms_iou: float = MODEL_CONFIG["nms_iou_threshold"]
    device: str = "cpu"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 20_000_000

    def __post_init__(self):
        object.__setattr__(self, "storage_dir", self.storage_dir.resolve())
        object.__setattr__(self, "database_path", (self.database_path or self.storage_dir / "acne-care.sqlite3").resolve())
        if not 0 < self.confidence_threshold < 1 or not 0 < self.nms_iou < 1 or self.image_size <= 0 or self.max_upload_bytes <= 0 or self.max_pixels <= 0:
            raise ValueError("invalid analysis settings")

    @classmethod
    def from_environment(cls):
        return cls(
            storage_dir=Path(os.environ.get("ACNE_STORAGE_DIR", str(cls.storage_dir))),
            database_path=Path(os.environ["ACNE_DATABASE_PATH"]) if "ACNE_DATABASE_PATH" in os.environ else None,
            weights_path=Path(os.environ.get("ACNE_WEIGHTS_PATH", str(cls.weights_path))),
            model_id=os.environ.get("ACNE_MODEL_ID", cls.model_id),
            expected_sha256=os.environ.get("ACNE_MODEL_SHA256", cls.expected_sha256),
            device=os.environ.get("ACNE_DEVICE", "cpu"),
        )
