"""Domain types shared by model and service adapters."""

from .detection import BoundingBox, Detection, GroundTruth

__all__ = ["BoundingBox", "Detection", "GroundTruth"]
