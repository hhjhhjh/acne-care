from __future__ import annotations

from collections import defaultdict
from typing import Any

from acne_care.domain import BoundingBox, Detection, GroundTruth
from acne_care.evaluation.detection_metrics import match_detections


def analyze_prediction_document(
    document: dict[str, Any],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    saved_threshold = float(document["confidence_threshold"])
    if confidence_threshold < saved_threshold:
        raise ValueError(
            "confidence threshold is below saved prediction threshold; "
            "rerun inference with a lower saved threshold"
        )

    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("prediction artifact samples must be a non-empty list")

    totals = {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
    by_size_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"ground_truths": 0, "detected": 0}
    )
    per_image = []

    for sample in samples:
        ground_truths = _ground_truths(sample)
        predictions = _predictions(sample, confidence_threshold)
        matches = match_detections(
            ground_truths,
            predictions,
            iou_threshold=iou_threshold,
        )
        detected_ground_truths = {pair[0] for pair in matches.matched_pairs}
        false_negative_sizes = {size: 0 for size in ("small", "medium", "large")}
        for ground_truth_index, (box, _) in enumerate(ground_truths):
            size = object_size(box)
            by_size_counts[size]["ground_truths"] += 1
            if ground_truth_index in detected_ground_truths:
                by_size_counts[size]["detected"] += 1
            else:
                false_negative_sizes[size] += 1

        true_positives = len(matches.matched_pairs)
        false_positives = len(matches.false_positive_indices)
        false_negatives = len(matches.false_negative_indices)
        totals["true_positives"] += true_positives
        totals["false_positives"] += false_positives
        totals["false_negatives"] += false_negatives
        per_image.append(
            {
                "image_path": str(sample["image_path"]),
                "ground_truth_count": len(ground_truths),
                "prediction_count": len(predictions),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "false_negatives_by_size": false_negative_sizes,
            }
        )

    return {
        "schema_version": 1,
        "model_id": document.get("model_id"),
        "split": document.get("split"),
        "saved_confidence_threshold": saved_threshold,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "images": len(samples),
        "ground_truths": totals["true_positives"] + totals["false_negatives"],
        "counts": totals,
        "precision": _divide(
            totals["true_positives"],
            totals["true_positives"] + totals["false_positives"],
        ),
        "recall": _divide(
            totals["true_positives"],
            totals["true_positives"] + totals["false_negatives"],
        ),
        "by_size": {
            size: _size_metrics(by_size_counts[size])
            for size in ("small", "medium", "large")
        },
        "iou_sensitivity": _iou_sensitivity(
            samples,
            confidence_threshold=confidence_threshold,
        ),
        "per_image": per_image,
    }


def object_size(box: BoundingBox) -> str:
    if box.area < 32**2:
        return "small"
    if box.area < 96**2:
        return "medium"
    return "large"


def summarize_condition_groups(
    per_image: list[dict[str, Any]], metric: str
) -> dict[str, dict[str, int | float]]:
    if not per_image:
        raise ValueError("at least one image is required for condition groups")
    if any(metric not in item for item in per_image):
        raise ValueError(f"condition metric is missing: {metric}")
    ordered = sorted(per_image, key=lambda item: item[metric])
    groups: dict[str, list[dict[str, Any]]] = {
        "low": [],
        "middle": [],
        "high": [],
    }
    names = tuple(groups)
    for index, item in enumerate(ordered):
        group_index = min(index * 3 // len(ordered), 2)
        groups[names[group_index]].append(item)
    return {
        name: _summarize_images(items, metric)
        for name, items in groups.items()
    }


def rank_error_images(
    per_image: list[dict[str, Any]], metric: str, *, limit: int = 10
) -> list[dict[str, Any]]:
    if metric not in {"false_positives", "false_negatives"}:
        raise ValueError(f"unsupported error metric: {metric}")
    return [
        {
            "image_path": item["image_path"],
            metric: item[metric],
            "ground_truth_count": item["ground_truth_count"],
            "brightness": item["brightness"],
            "sharpness": item["sharpness"],
        }
        for item in sorted(per_image, key=lambda item: item[metric], reverse=True)[:limit]
    ]


def _iou_sensitivity(
    samples: list[dict[str, Any]],
    *,
    confidence_threshold: float,
) -> dict[str, dict[str, int | float]]:
    results = {}
    for iou_threshold in (0.3, 0.5, 0.75):
        ground_truth_count = 0
        detected = 0
        for sample in samples:
            ground_truths = _ground_truths(sample)
            matches = match_detections(
                ground_truths,
                _predictions(sample, confidence_threshold),
                iou_threshold=iou_threshold,
            )
            ground_truth_count += len(ground_truths)
            detected += len(matches.matched_pairs)
        results[f"{iou_threshold:.2f}"] = {
            "ground_truths": ground_truth_count,
            "detected": detected,
            "recall": _divide(detected, ground_truth_count),
        }
    return results


def _ground_truths(sample: dict[str, Any]) -> tuple[GroundTruth, ...]:
    return tuple(
        (BoundingBox(*item["box"]), int(item["class_id"]))
        for item in sample["ground_truths"]
    )


def _predictions(
    sample: dict[str, Any], confidence_threshold: float
) -> tuple[Detection, ...]:
    return tuple(
        Detection(
            BoundingBox(*item["box"]),
            confidence=float(item["confidence"]),
            class_id=int(item["class_id"]),
        )
        for item in sample["predictions"]
        if float(item["confidence"]) >= confidence_threshold
    )


def _size_metrics(counts: dict[str, int]) -> dict[str, int | float]:
    ground_truths = counts["ground_truths"]
    detected = counts["detected"]
    return {
        "ground_truths": ground_truths,
        "detected": detected,
        "missed": ground_truths - detected,
        "recall": _divide(detected, ground_truths),
    }


def _summarize_images(
    items: list[dict[str, Any]], metric: str
) -> dict[str, int | float]:
    true_positives = sum(item["true_positives"] for item in items)
    false_positives = sum(item["false_positives"] for item in items)
    false_negatives = sum(item["false_negatives"] for item in items)
    return {
        "images": len(items),
        "ground_truths": sum(item["ground_truth_count"] for item in items),
        f"{metric}_minimum": min(item[metric] for item in items),
        f"{metric}_maximum": max(item[metric] for item in items),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": _divide(true_positives, true_positives + false_positives),
        "recall": _divide(true_positives, true_positives + false_negatives),
    }


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
