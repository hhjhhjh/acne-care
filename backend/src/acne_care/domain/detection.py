from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A box in pixel coordinates of the image being analyzed."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        coordinates = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(isfinite(coordinate) for coordinate in coordinates):
            raise ValueError("bounding box coordinates must be finite")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
        )

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def validate_image_bounds(self, *, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        if (
            self.x_min < 0
            or self.y_min < 0
            or self.x_max > width
            or self.y_max > height
        ):
            raise ValueError("bounding box must stay within image bounds")


@dataclass(frozen=True, slots=True)
class Detection:
    """One model detection with a normalized confidence value."""

    box: BoundingBox
    confidence: float
    class_id: int

    def __post_init__(self) -> None:
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.class_id < 0:
            raise ValueError("class_id must be zero or greater")


GroundTruth = tuple[BoundingBox, int]
