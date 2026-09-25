from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from acne_care.dataset.yolo import read_yolo_ground_truths
from acne_care.domain import BoundingBox, Detection
from acne_care.evaluation.detection_metrics import (
    ThresholdMetrics,
    evaluate_thresholds,
    select_best_threshold,
)


CLASS_ID = 0


def main() -> None:
    arguments = parse_arguments()
    image_paths = read_manifest_images(arguments.image_manifest)
    if not image_paths:
        raise ValueError("test manifest contains no images")

    model = YOLO(str(arguments.weights))
    validation = model.val(
        data=str(arguments.data),
        split=arguments.split,
        imgsz=arguments.image_size,
        batch=arguments.batch_size,
        device=arguments.device,
        plots=False,
        verbose=False,
    )

    minimum_confidence = min(
        (arguments.confidence_threshold, *(arguments.thresholds or ()))
    )
    synchronize_device(arguments.device)
    inference_started_at = time.perf_counter()
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=arguments.image_size,
        conf=minimum_confidence,
        device=arguments.device,
        batch=arguments.batch_size,
        verbose=False,
        stream=False,
    )
    synchronize_device(arguments.device)
    inference_seconds = time.perf_counter() - inference_started_at
    samples = []
    for image_path, result in zip(image_paths, results, strict=True):
        ground_truths = read_yolo_ground_truths(
            label_path_for(image_path),
            image_width=result.orig_shape[1],
            image_height=result.orig_shape[0],
        )
        detections = tuple(
            Detection(
                box=BoundingBox(*(float(value) for value in box_values)),
                confidence=float(confidence),
                class_id=int(class_id),
            )
            for box_values, confidence, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            )
            if int(class_id) == CLASS_ID
        )
        samples.append((ground_truths, detections))

    primary_metrics = evaluate_thresholds(
        samples,
        thresholds=(arguments.confidence_threshold,),
        iou_threshold=arguments.iou_threshold,
    )[0]
    sweep_results = evaluate_thresholds(
        samples,
        thresholds=arguments.thresholds,
        iou_threshold=arguments.iou_threshold,
    ) if arguments.thresholds else ()

    result_document = {
        "model": {
            "id": "yolo11n-acne-seed42",
            "weights": str(arguments.weights),
            "weights_sha256": file_sha256(arguments.weights),
        },
        "runtime": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "device": arguments.device,
        },
        "dataset": {
            "data_yaml": str(arguments.data),
            "split": arguments.split,
            "image_manifest": str(arguments.image_manifest),
            "images_evaluated": len(image_paths),
        },
        "evaluation": {
            "confidence_threshold": arguments.confidence_threshold,
            "iou_threshold": arguments.iou_threshold,
            "counts": asdict(primary_metrics.counts),
            "precision": primary_metrics.precision,
            "recall": primary_metrics.recall,
            "f1": primary_metrics.f1,
            "count_mae": primary_metrics.count_mae,
            "coco": {
                "mAP50-95": float(validation.box.map),
                "mAP50": float(validation.box.map50),
                "mAP75": float(validation.box.map75),
            },
            "prediction_pipeline_seconds_total": inference_seconds,
            "prediction_pipeline_seconds_per_image": inference_seconds
            / len(image_paths),
        },
    }
    if sweep_results:
        result_document["threshold_sweep"] = {
            "selection_rule": "maximum F1, then minimum count MAE, then maximum recall",
            "selected": threshold_result_to_dict(
                select_best_threshold(sweep_results)
            ),
            "results": [
                threshold_result_to_dict(item) for item in sweep_results
            ],
        }
    serialized = json.dumps(result_document, ensure_ascii=False, indent=2)
    print(serialized)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO11 acne model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--image-manifest",
        "--test-manifest",
        dest="image_manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", nargs="*", type=float)
    return parser.parse_args()


def read_manifest_images(path: Path) -> list[Path]:
    return [
        Path(line)
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip())
    ]


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        image_directory_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"image path has no images directory: {image_path}") from error
    parts[image_directory_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize_device(device: str) -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def threshold_result_to_dict(result: ThresholdMetrics) -> dict[str, object]:
    return {
        "threshold": result.threshold,
        "counts": asdict(result.counts),
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "count_mae": result.count_mae,
    }


if __name__ == "__main__":
    main()
