from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from acne_care.domain import Detection


COLUMNS = ("id", "captured_on", "created_at", "width", "height", "model_id", "weights_sha256",
           "confidence_threshold", "image_size", "device", "nms_iou", "inference_seconds", "original_path", "annotated_path")
SELECT = "SELECT a.*, (SELECT COUNT(*) FROM detections d WHERE d.analysis_id=a.id) AS count FROM analyses a"


class SQLiteAnalysisRepository:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def _connection(self):
        with closing(sqlite3.connect(self.path, timeout=15)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                yield connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError("unsupported analysis database version")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY, captured_on TEXT NOT NULL, created_at TEXT NOT NULL,
                    width INTEGER NOT NULL CHECK(width>0), height INTEGER NOT NULL CHECK(height>0),
                    model_id TEXT NOT NULL, weights_sha256 TEXT NOT NULL,
                    confidence_threshold REAL NOT NULL CHECK(confidence_threshold BETWEEN 0 AND 1),
                    image_size INTEGER NOT NULL, device TEXT NOT NULL, nms_iou REAL NOT NULL,
                    inference_seconds REAL NOT NULL, original_path TEXT NOT NULL, annotated_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS detections (
                    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL, class_id INTEGER NOT NULL CHECK(class_id=0),
                    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
                    x_min REAL NOT NULL, y_min REAL NOT NULL, x_max REAL NOT NULL, y_max REAL NOT NULL,
                    PRIMARY KEY(analysis_id,ordinal), CHECK(x_max>x_min), CHECK(y_max>y_min)
                );
                CREATE INDEX IF NOT EXISTS analyses_capture_date ON analyses(captured_on DESC, created_at DESC);
                PRAGMA user_version=1;
            """)

    def insert(self, record: dict[str, Any], detections: tuple[Detection, ...]) -> None:
        with self._connection() as connection:
            connection.execute(f"INSERT INTO analyses ({','.join(COLUMNS)}) VALUES ({','.join('?' for _ in COLUMNS)})",
                               [record[column] for column in COLUMNS])
            connection.executemany("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?)", [
                (record["id"], index, detection.class_id, detection.confidence,
                 detection.box.x_min, detection.box.y_min, detection.box.x_max, detection.box.y_max)
                for index, detection in enumerate(detections)
            ])

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(SELECT + " WHERE a.id=?", (analysis_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["detections"] = [
                {"class_id": item["class_id"], "confidence": item["confidence"],
                 "box": {key: item[key] for key in ("x_min", "y_min", "x_max", "y_max")}}
                for item in connection.execute("SELECT * FROM detections WHERE analysis_id=? ORDER BY ordinal", (analysis_id,))
            ]
            return result

    def list(self, limit: int, offset: int) -> dict[str, Any]:
        with self._connection() as connection:
            total = connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            rows = connection.execute(SELECT + " ORDER BY captured_on DESC, created_at DESC, id DESC LIMIT ? OFFSET ?", (limit, offset))
            return {"items": [dict(row) for row in rows], "total": total, "limit": limit, "offset": offset}
