from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

from acne_care.domain import Detection, GroundTruth


def serialize_sample(
    *,
    image_path: Path,
    image_width: int,
    image_height: int,
    ground_truths: Sequence[GroundTruth],
    predictions: Sequence[Detection],
    confidence_threshold: float,
) -> dict[str, object]:
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    return {
        "image_path": str(image_path),
        "image_width": image_width,
        "image_height": image_height,
        "ground_truths": [
            {"box": box_values(box), "class_id": class_id}
            for box, class_id in ground_truths
        ],
        "predictions": [
            {
                "box": box_values(prediction.box),
                "confidence": prediction.confidence,
                "class_id": prediction.class_id,
            }
            for prediction in predictions
            if prediction.confidence >= confidence_threshold
        ],
    }


def box_values(box) -> list[float]:
    return [box.x_min, box.y_min, box.x_max, box.y_max]


def save_prediction_artifact(
    output_path: Path,
    *,
    model_id: str,
    split: str,
    confidence_threshold: float,
    iou_threshold: float,
    samples: list[dict[str, object]],
) -> None:
    document = {
        "schema_version": 1,
        "model_id": model_id,
        "split": split,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
