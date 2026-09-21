"""Model-independent evaluation helpers."""

from .detection_metrics import DetectionMetricCounts, evaluate_detections

__all__ = ["DetectionMetricCounts", "evaluate_detections"]

