from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from acne_care.domain import BoundingBox, Detection
from acne_care.evaluation.detection_metrics import match_detections


def main() -> None:
    arguments = parse_arguments()
    legacy = read_samples(arguments.legacy_predictions)
    yolo11 = read_samples(arguments.yolo11_predictions)
    summary = json.loads(arguments.summary.read_text(encoding="utf-8"))
    ranked_paths = []
    for ranking in summary["ranking"].values():
        for item in ranking:
            image_name = Path(item["image_path"]).name
            image_path = next(
                path for path in legacy if Path(path).name == image_name
            )
            if image_path not in ranked_paths:
                ranked_paths.append(image_path)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    for image_path in ranked_paths[: arguments.limit]:
        first = render_panel(legacy[image_path], "legacy", arguments.iou_threshold)
        second = render_panel(yolo11[image_path], "YOLO11n", arguments.iou_threshold)
        comparison = cv2.hconcat((first, second))
        output_path = arguments.output_directory / Path(image_path).name
        if not cv2.imwrite(str(output_path), comparison):
            raise ValueError(f"failed to write overlay: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render model error overlays")
    parser.add_argument("--legacy-predictions", type=Path, required=True)
    parser.add_argument("--yolo11-predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def read_samples(path: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {sample["image_path"]: sample for sample in document["samples"]}


def render_panel(sample: dict[str, Any], title: str, iou_threshold: float):
    image = cv2.imread(sample["image_path"])
    if image is None:
        raise ValueError(f"failed to read image: {sample['image_path']}")
    height, width = image.shape[:2]
    coordinate_width = sample["image_width"]
    coordinate_height = sample["image_height"]
    ground_truths = tuple(
        (BoundingBox(*item["box"]), int(item["class_id"]))
        for item in sample["ground_truths"]
    )
    predictions = tuple(
        Detection(
            BoundingBox(*item["box"]),
            confidence=float(item["confidence"]),
            class_id=int(item["class_id"]),
        )
        for item in sample["predictions"]
    )
    matches = match_detections(
        ground_truths, predictions, iou_threshold=iou_threshold
    )
    matched_predictions = {pair[1] for pair in matches.matched_pairs}
    for prediction_index, prediction in enumerate(predictions):
        color = (0, 180, 0) if prediction_index in matched_predictions else (0, 0, 255)
        draw_box(
            image,
            prediction.box,
            color,
            width / coordinate_width,
            height / coordinate_height,
        )
    for ground_truth_index in matches.false_negative_indices:
        draw_box(
            image,
            ground_truths[ground_truth_index][0],
            (255, 180, 0),
            width / coordinate_width,
            height / coordinate_height,
        )
    cv2.rectangle(image, (0, 0), (210, 34), (20, 20, 20), -1)
    cv2.putText(
        image,
        f"{title}  green=TP red=FP cyan=FN",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def draw_box(image, box: BoundingBox, color, scale_x: float, scale_y: float) -> None:
    first = (round(box.x_min * scale_x), round(box.y_min * scale_y))
    second = (round(box.x_max * scale_x), round(box.y_max * scale_y))
    cv2.rectangle(image, first, second, color, 2)


if __name__ == "__main__":
    main()
