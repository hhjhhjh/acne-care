from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from acne_care.domain import BoundingBox, Detection, GroundTruth


@dataclass(frozen=True, slots=True)
class DetectionMetricCounts:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def __add__(self, other: DetectionMetricCounts) -> DetectionMetricCounts:
        if not isinstance(other, DetectionMetricCounts):
            return NotImplemented
        return DetectionMetricCounts(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )


def evaluate_detections(
    ground_truths: Sequence[GroundTruth],
    predictions: Sequence[Detection],
    *,
    iou_threshold: float,
) -> DetectionMetricCounts:
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")

    matched_ground_truths: set[int] = set()
    true_positives = 0
    false_positives = 0

    for prediction in sorted(
        predictions,
        key=lambda detection: detection.confidence,
        reverse=True,
    ):
        best_index: int | None = None
        best_iou = 0.0
        for index, (truth_box, truth_class_id) in enumerate(ground_truths):
            if index in matched_ground_truths:
                continue
            if truth_class_id != prediction.class_id:
                continue
            overlap = intersection_over_union(prediction.box, truth_box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched_ground_truths.add(best_index)
            true_positives += 1
        else:
            false_positives += 1

    return DetectionMetricCounts(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=len(ground_truths) - len(matched_ground_truths),
    )


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    intersection_width = max(
        0.0,
        min(first.x_max, second.x_max) - max(first.x_min, second.x_min),
    )
    intersection_height = max(
        0.0,
        min(first.y_max, second.y_max) - max(first.y_min, second.y_min),
    )
    intersection_area = intersection_width * intersection_height
    union_area = first.area + second.area - intersection_area
    return intersection_area / union_area if union_area else 0.0
