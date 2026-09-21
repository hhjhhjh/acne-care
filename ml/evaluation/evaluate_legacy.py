from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import keras
import keras_cv
import numpy as np
import tensorflow as tf

from acne_care.dataset.yolo import read_yolo_ground_truths
from acne_care.domain import BoundingBox, Detection
from acne_care.evaluation.detection_metrics import (
    DetectionMetricCounts,
    evaluate_detections,
)


IMAGE_SIZE = 640
CLASS_ID = 0


def main() -> None:
    arguments = parse_arguments()
    excluded_paths = read_exclusions(arguments.exclude_file)
    image_paths = sorted((arguments.dataset_root / "test" / "images").glob("*.jpg"))
    selected_paths = [
        path
        for path in image_paths
        if path.relative_to(arguments.dataset_root).as_posix() not in excluded_paths
    ]
    if not selected_paths:
        raise ValueError("no test images remain after applying exclusions")

    maximum_ground_truth_boxes = max(
        len(
            read_yolo_ground_truths(
                arguments.dataset_root
                / "test"
                / "labels"
                / f"{image_path.stem}.txt",
                image_width=IMAGE_SIZE,
                image_height=IMAGE_SIZE,
            )
        )
        for image_path in selected_paths
    )

    model = load_model(arguments.weights)
    coco_metric = keras_cv.metrics.BoxCOCOMetrics(
        bounding_box_format="xyxy",
        evaluate_freq=len(selected_paths) + 1,
    )
    aggregate_counts = DetectionMetricCounts()
    inference_seconds = 0.0

    for batch_start in range(0, len(selected_paths), arguments.batch_size):
        batch_paths = selected_paths[batch_start : batch_start + arguments.batch_size]
        images, ground_truth_batches = load_batch(arguments.dataset_root, batch_paths)
        started_at = time.perf_counter()
        predictions = model.predict(images, verbose=0)
        inference_seconds += time.perf_counter() - started_at

        y_true = pad_ground_truths(
            ground_truth_batches,
            maximum_boxes=maximum_ground_truth_boxes,
        )
        coco_metric.update_state(y_true, predictions)

        for index, ground_truths in enumerate(ground_truth_batches):
            detections = prediction_to_detections(
                predictions,
                batch_index=index,
                confidence_threshold=arguments.confidence_threshold,
            )
            aggregate_counts += evaluate_detections(
                ground_truths,
                detections,
                iou_threshold=arguments.iou_threshold,
            )

    coco_values = {
        key: float(value.numpy())
        for key, value in coco_metric.result(force=True).items()
    }
    result = {
        "model": {
            "id": "acne-cv-legacy-keras-cv-yolov8-xs",
            "weights": str(arguments.weights),
            "weights_sha256": file_sha256(arguments.weights),
        },
        "runtime": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "keras_cv": keras_cv.__version__,
        },
        "dataset": {
            "root": str(arguments.dataset_root),
            "test_images_total": len(image_paths),
            "test_images_evaluated": len(selected_paths),
            "excluded_paths": sorted(excluded_paths),
        },
        "evaluation": {
            "confidence_threshold": arguments.confidence_threshold,
            "iou_threshold": arguments.iou_threshold,
            "counts": asdict(aggregate_counts),
            "precision": aggregate_counts.precision,
            "recall": aggregate_counts.recall,
            "f1": aggregate_counts.f1,
            "coco": coco_values,
            "inference_seconds_total": inference_seconds,
            "inference_seconds_per_image": inference_seconds / len(selected_paths),
        },
    }

    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the legacy acne-CV model")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--exclude-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_model(weights_path: Path):
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone",
        include_rescaling=True,
    )
    model = keras_cv.models.YOLOV8Detector(
        num_classes=1,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=5,
    )
    model.load_weights(weights_path)
    return model


def load_batch(
    dataset_root: Path,
    image_paths: list[Path],
) -> tuple[tf.Tensor, list[tuple[tuple[BoundingBox, int], ...]]]:
    images: list[tf.Tensor] = []
    ground_truth_batches: list[tuple[tuple[BoundingBox, int], ...]] = []
    for image_path in image_paths:
        image = tf.io.decode_jpeg(tf.io.read_file(str(image_path)), channels=3)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(tf.cast(image, tf.float32))
        label_path = (
            dataset_root / "test" / "labels" / f"{image_path.stem}.txt"
        )
        ground_truth_batches.append(
            read_yolo_ground_truths(
                label_path,
                image_width=IMAGE_SIZE,
                image_height=IMAGE_SIZE,
            )
        )
    return tf.stack(images), ground_truth_batches


def pad_ground_truths(
    batches: list[tuple[tuple[BoundingBox, int], ...]],
    *,
    maximum_boxes: int,
) -> dict[str, tf.Tensor]:
    if any(len(batch) > maximum_boxes for batch in batches):
        raise ValueError("maximum_boxes is smaller than a ground-truth batch")
    boxes = np.full((len(batches), maximum_boxes, 4), -1.0, dtype=np.float32)
    classes = np.full((len(batches), maximum_boxes), -1.0, dtype=np.float32)
    for batch_index, ground_truths in enumerate(batches):
        for box_index, (box, class_id) in enumerate(ground_truths):
            boxes[batch_index, box_index] = [
                box.x_min,
                box.y_min,
                box.x_max,
                box.y_max,
            ]
            classes[batch_index, box_index] = class_id
    return {"boxes": tf.constant(boxes), "classes": tf.constant(classes)}


def prediction_to_detections(
    predictions: dict[str, np.ndarray],
    *,
    batch_index: int,
    confidence_threshold: float,
) -> tuple[Detection, ...]:
    detections: list[Detection] = []
    for box_values, class_value, confidence_value in zip(
        predictions["boxes"][batch_index],
        predictions["classes"][batch_index],
        predictions["confidence"][batch_index],
    ):
        confidence = float(confidence_value)
        class_id = int(class_value)
        if confidence < confidence_threshold or class_id != CLASS_ID:
            continue
        box = BoundingBox(*(float(value) for value in box_values))
        detections.append(
            Detection(box=box, confidence=confidence, class_id=class_id)
        )
    return tuple(detections)


def read_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
