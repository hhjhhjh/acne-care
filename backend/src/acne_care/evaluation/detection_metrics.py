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


@dataclass(frozen=True, slots=True)
class DetectionMatches:
    matched_pairs: tuple[tuple[int, int, float], ...]
    false_positive_indices: tuple[int, ...]
    false_negative_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    threshold: float
    counts: DetectionMetricCounts
    image_count: int
    total_absolute_count_error: int

    @property
    def true_positives(self) -> int:
        return self.counts.true_positives

    @property
    def false_positives(self) -> int:
        return self.counts.false_positives

    @property
    def false_negatives(self) -> int:
        return self.counts.false_negatives

    @property
    def precision(self) -> float:
        return self.counts.precision

    @property
    def recall(self) -> float:
        return self.counts.recall

    @property
    def f1(self) -> float:
        return self.counts.f1

    @property
    def count_mae(self) -> float:
        return self.total_absolute_count_error / self.image_count


def evaluate_detections(
    ground_truths: Sequence[GroundTruth],
    predictions: Sequence[Detection],
    *,
    iou_threshold: float,
) -> DetectionMetricCounts:
    matches = match_detections(
        ground_truths,
        predictions,
        iou_threshold=iou_threshold,
    )
    return DetectionMetricCounts(
        true_positives=len(matches.matched_pairs),
        false_positives=len(matches.false_positive_indices),
        false_negatives=len(matches.false_negative_indices),
    )


def match_detections(
    ground_truths: Sequence[GroundTruth],
    predictions: Sequence[Detection],
    *,
    iou_threshold: float,
) -> DetectionMatches:
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")

    matched_ground_truths: set[int] = set()
    matched_pairs: list[tuple[int, int, float]] = []
    false_positive_indices: list[int] = []

    indexed_predictions = sorted(
        enumerate(predictions),
        key=lambda item: item[1].confidence,
        reverse=True,
    )
    for prediction_index, prediction in indexed_predictions:
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
            matched_pairs.append((best_index, prediction_index, best_iou))
        else:
            false_positive_indices.append(prediction_index)

    return DetectionMatches(
        matched_pairs=tuple(matched_pairs),
        false_positive_indices=tuple(sorted(false_positive_indices)),
        false_negative_indices=tuple(
            index
            for index in range(len(ground_truths))
            if index not in matched_ground_truths
        ),
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


def evaluate_thresholds(
    samples: Sequence[tuple[Sequence[GroundTruth], Sequence[Detection]]],
    *,
    thresholds: Sequence[float],
    iou_threshold: float,
) -> tuple[ThresholdMetrics, ...]:
    if not samples:
        raise ValueError("at least one evaluation sample is required")

    results: list[ThresholdMetrics] = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("confidence thresholds must be between 0 and 1")
        counts = DetectionMetricCounts()
        total_absolute_count_error = 0
        for ground_truths, predictions in samples:
            selected_predictions = tuple(
                prediction
                for prediction in predictions
                if prediction.confidence >= threshold
            )
            counts += evaluate_detections(
                ground_truths,
                selected_predictions,
                iou_threshold=iou_threshold,
            )
            total_absolute_count_error += abs(
                len(selected_predictions) - len(ground_truths)
            )
        results.append(
            ThresholdMetrics(
                threshold=float(threshold),
                counts=counts,
                image_count=len(samples),
                total_absolute_count_error=total_absolute_count_error,
            )
        )
    return tuple(results)


def select_best_threshold(
    results: Sequence[ThresholdMetrics],
) -> ThresholdMetrics:
    if not results:
        raise ValueError("at least one threshold result is required")
    return max(
        results,
        key=lambda result: (
            result.f1,
            -result.count_mae,
            result.recall,
            result.threshold,
        ),
    )
