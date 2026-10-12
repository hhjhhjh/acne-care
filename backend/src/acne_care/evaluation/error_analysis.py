from __future__ import annotations

from collections import defaultdict
from typing import Any

from acne_care.domain import BoundingBox, Detection, GroundTruth
from acne_care.evaluation.detection_metrics import match_detections


def compare_prediction_documents(
    legacy_document: dict[str, Any],
    yolo11_document: dict[str, Any],
) -> dict[str, Any]:
    _validate_documents(legacy_document, yolo11_document)
    legacy_samples = _samples_by_path(legacy_document)
    yolo11_samples = _samples_by_path(yolo11_document)
    if legacy_samples.keys() != yolo11_samples.keys():
        raise ValueError("prediction artifacts must contain the same image paths")

    outcomes = {name: 0 for name in outcome_names()}
    false_positives = {"legacy": 0, "yolo11": 0}
    by_size_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"ground_truths": 0, "legacy_detected": 0, "yolo11_detected": 0}
    )
    per_image = []
    iou_threshold = float(legacy_document["iou_threshold"])

    for image_path in sorted(legacy_samples):
        legacy_sample = legacy_samples[image_path]
        yolo11_sample = yolo11_samples[image_path]
        _validate_sample_pair(legacy_sample, yolo11_sample)
        ground_truths = _ground_truths(legacy_sample)
        legacy_predictions = _predictions(legacy_sample)
        yolo11_predictions = _predictions(yolo11_sample)
        legacy_matches = match_detections(
            ground_truths, legacy_predictions, iou_threshold=iou_threshold
        )
        yolo11_matches = match_detections(
            ground_truths, yolo11_predictions, iou_threshold=iou_threshold
        )
        legacy_detected = {pair[0] for pair in legacy_matches.matched_pairs}
        yolo11_detected = {pair[0] for pair in yolo11_matches.matched_pairs}
        image_outcomes = {name: 0 for name in outcome_names()}

        for ground_truth_index, (box, _) in enumerate(ground_truths):
            outcome = outcome_for(
                ground_truth_index in legacy_detected,
                ground_truth_index in yolo11_detected,
            )
            outcomes[outcome] += 1
            image_outcomes[outcome] += 1
            size = object_size(box)
            by_size_counts[size]["ground_truths"] += 1
            by_size_counts[size]["legacy_detected"] += int(
                ground_truth_index in legacy_detected
            )
            by_size_counts[size]["yolo11_detected"] += int(
                ground_truth_index in yolo11_detected
            )

        legacy_fp = len(legacy_matches.false_positive_indices)
        yolo11_fp = len(yolo11_matches.false_positive_indices)
        false_positives["legacy"] += legacy_fp
        false_positives["yolo11"] += yolo11_fp
        per_image.append(
            {
                "image_path": image_path,
                "ground_truth_count": len(ground_truths),
                "outcomes": image_outcomes,
                "false_positives": {"legacy": legacy_fp, "yolo11": yolo11_fp},
            }
        )

    return {
        "images": len(per_image),
        "ground_truths": sum(outcomes.values()),
        "outcomes": outcomes,
        "false_positives": false_positives,
        "iou_sensitivity": iou_sensitivity(legacy_samples, yolo11_samples),
        "by_size": {
            size: size_metrics(by_size_counts[size])
            for size in ("small", "medium", "large")
        },
        "per_image": per_image,
    }


def iou_sensitivity(
    legacy_samples: dict[str, dict[str, Any]],
    yolo11_samples: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    results = {}
    for threshold in (0.3, 0.5, 0.75):
        total = 0
        detected = {"legacy": 0, "yolo11": 0}
        for image_path in legacy_samples:
            ground_truths = _ground_truths(legacy_samples[image_path])
            total += len(ground_truths)
            for model_name, samples in (
                ("legacy", legacy_samples),
                ("yolo11", yolo11_samples),
            ):
                matches = match_detections(
                    ground_truths,
                    _predictions(samples[image_path]),
                    iou_threshold=threshold,
                )
                detected[model_name] += len(matches.matched_pairs)
        results[f"{threshold:.2f}"] = {
            "ground_truths": total,
            "legacy_recall": detected["legacy"] / total if total else 0.0,
            "yolo11_recall": detected["yolo11"] / total if total else 0.0,
        }
    return results


def outcome_names() -> tuple[str, ...]:
    return ("both_detected", "yolo11_only", "legacy_only", "both_missed")


def outcome_for(legacy_detected: bool, yolo11_detected: bool) -> str:
    if legacy_detected and yolo11_detected:
        return "both_detected"
    if yolo11_detected:
        return "yolo11_only"
    if legacy_detected:
        return "legacy_only"
    return "both_missed"


def object_size(box: BoundingBox) -> str:
    if box.area < 32**2:
        return "small"
    if box.area < 96**2:
        return "medium"
    return "large"


def size_metrics(counts: dict[str, int]) -> dict[str, int | float]:
    total = counts["ground_truths"]
    return {
        **counts,
        "legacy_recall": counts["legacy_detected"] / total if total else 0.0,
        "yolo11_recall": counts["yolo11_detected"] / total if total else 0.0,
    }


def _samples_by_path(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    samples = document.get("samples")
    if not isinstance(samples, list):
        raise ValueError("prediction artifact samples must be a list")
    indexed = {str(sample["image_path"]): sample for sample in samples}
    if len(indexed) != len(samples):
        raise ValueError("prediction artifact contains duplicate image paths")
    return indexed


def _validate_documents(first: dict[str, Any], second: dict[str, Any]) -> None:
    for field in ("schema_version", "split", "iou_threshold"):
        if first.get(field) != second.get(field):
            raise ValueError(f"prediction artifact field differs: {field}")


def _validate_sample_pair(first: dict[str, Any], second: dict[str, Any]) -> None:
    for field in ("image_width", "image_height", "ground_truths"):
        if first.get(field) != second.get(field):
            raise ValueError(f"sample field differs: {field}")


def _ground_truths(sample: dict[str, Any]) -> tuple[GroundTruth, ...]:
    return tuple(
        (BoundingBox(*item["box"]), int(item["class_id"]))
        for item in sample["ground_truths"]
    )


def _predictions(sample: dict[str, Any]) -> tuple[Detection, ...]:
    return tuple(
        Detection(
            BoundingBox(*item["box"]),
            confidence=float(item["confidence"]),
            class_id=int(item["class_id"]),
        )
        for item in sample["predictions"]
    )
