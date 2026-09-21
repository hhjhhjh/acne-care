from __future__ import annotations

from pathlib import Path

from acne_care.domain import BoundingBox, GroundTruth


def read_yolo_ground_truths(
    label_path: Path,
    *,
    image_width: int,
    image_height: int,
) -> tuple[GroundTruth, ...]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    ground_truths: list[GroundTruth] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        location = f"{label_path.name}:{line_number}"
        if len(fields) != 5:
            raise ValueError(f"{location} must contain exactly five values")
        try:
            class_value = float(fields[0])
            x_center, y_center, width, height = (
                float(value) for value in fields[1:]
            )
        except ValueError as error:
            raise ValueError(f"{location} contains a non-numeric value") from error

        if not class_value.is_integer() or class_value < 0:
            raise ValueError(f"{location} class_id must be a non-negative integer")
        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            raise ValueError(f"{location} contains invalid normalized coordinates")

        box = BoundingBox(
            x_min=(x_center - width / 2) * image_width,
            y_min=(y_center - height / 2) * image_height,
            x_max=(x_center + width / 2) * image_width,
            y_max=(y_center + height / 2) * image_height,
        )
        box.validate_image_bounds(width=image_width, height=image_height)
        ground_truths.append((box, int(class_value)))

    return tuple(ground_truths)
