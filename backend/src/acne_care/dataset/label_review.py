from __future__ import annotations

from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any

from acne_care.dataset.audit import IMAGE_SUFFIXES, SPLIT_ALIASES, audit_yolo_dataset


def build_label_review(
    root: Path,
    *,
    expected_class_ids: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    root = root.resolve()
    items: list[dict[str, Any]] = []
    for canonical_split, aliases in SPLIT_ALIASES.items():
        split_name = next((name for name in aliases if (root / name).is_dir()), None)
        if split_name is None:
            continue
        image_dir = root / split_name / "images"
        label_dir = root / split_name / "labels"
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            annotations, hints, details = inspect_label(
                label_path,
                expected_class_ids=expected_class_ids,
            )
            items.append(
                {
                    "split": canonical_split,
                    "image_path": image_path.relative_to(root).as_posix(),
                    "label_path": label_path.relative_to(root).as_posix(),
                    "annotation_count": len(annotations),
                    "annotations": annotations,
                    "review_hints": hints,
                    "details": details,
                }
            )

    add_statistical_hints(items)
    audit = audit_yolo_dataset(root)
    hint_counts = Counter(
        hint for item in items for hint in item["review_hints"]
    )
    return {
        "schema_version": 1,
        "dataset_root": str(root),
        "summary": {
            "images": len(items),
            "annotations": sum(item["annotation_count"] for item in items),
            "images_by_split": dict(audit.image_counts),
            "labels_by_split": dict(audit.label_counts),
            "class_ids": list(audit.class_ids),
            "structural_error_count": len(audit.errors),
            "review_hint_counts": dict(sorted(hint_counts.items())),
        },
        "structural_errors": list(audit.errors),
        "items": items,
    }


def inspect_label(
    label_path: Path,
    *,
    expected_class_ids: tuple[int, ...],
) -> tuple[list[dict[str, int | float]], list[str], list[str]]:
    annotations: list[dict[str, int | float]] = []
    hints: set[str] = set()
    details: list[str] = []
    if not label_path.is_file():
        return [], ["missing_label"], ["matching label file is missing"]

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [], ["invalid_label"], ["label file is not valid UTF-8"]

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            hints.add("invalid_label")
            details.append(f"line {line_number} does not contain five values")
            continue
        try:
            class_value = float(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError:
            hints.add("invalid_label")
            details.append(f"line {line_number} contains a non-numeric value")
            continue
        if not class_value.is_integer() or class_value < 0:
            hints.add("invalid_label")
            details.append(f"line {line_number} has an invalid class id")
            continue
        class_id = int(class_value)
        x_center, y_center, width, height = coordinates
        if not _coordinates_are_valid(coordinates):
            hints.add("invalid_label")
            details.append(f"line {line_number} has coordinates outside the image")
            continue
        if class_id not in expected_class_ids:
            hints.add("unexpected_class")
            details.append(f"line {line_number} uses unexpected class {class_id}")
        annotations.append(
            {
                "line_number": line_number,
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "area_ratio": width * height,
            }
        )

    if not annotations:
        hints.add("empty_label")
    normalized_boxes = [
        (
            annotation["class_id"],
            annotation["x_center"],
            annotation["y_center"],
            annotation["width"],
            annotation["height"],
        )
        for annotation in annotations
    ]
    if len(normalized_boxes) != len(set(normalized_boxes)):
        hints.add("duplicate_box")
        details.append("the label contains an identical duplicate box")
    return annotations, sorted(hints), details


def add_statistical_hints(items: list[dict[str, Any]]) -> None:
    all_areas = sorted(
        float(annotation["area_ratio"])
        for item in items
        for annotation in item["annotations"]
    )
    if len(all_areas) >= 100:
        small_limit = percentile(all_areas, 0.01)
        large_limit = percentile(all_areas, 0.99)
        for item in items:
            areas = [
                float(annotation["area_ratio"])
                for annotation in item["annotations"]
            ]
            if any(area <= small_limit for area in areas):
                item["review_hints"].append("very_small_box")
            if any(area >= large_limit for area in areas):
                item["review_hints"].append("very_large_box")

    for split in SPLIT_ALIASES:
        split_items = [item for item in items if item["split"] == split]
        counts = sorted(item["annotation_count"] for item in split_items)
        if len(counts) < 4:
            continue
        upper_limit = percentile(counts, 0.75) + 1.5 * (
            percentile(counts, 0.75) - percentile(counts, 0.25)
        )
        for item in split_items:
            if item["annotation_count"] > upper_limit:
                item["review_hints"].append("high_annotation_count")
    for item in items:
        item["review_hints"] = sorted(set(item["review_hints"]))


def percentile(values: list[float] | list[int], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _coordinates_are_valid(coordinates: tuple[float, ...]) -> bool:
    if not all(isfinite(value) for value in coordinates):
        return False
    x_center, y_center, width, height = coordinates
    return (
        0 <= x_center <= 1
        and 0 <= y_center <= 1
        and 0 < width <= 1
        and 0 < height <= 1
        and 0 <= x_center - width / 2
        and x_center + width / 2 <= 1
        and 0 <= y_center - height / 2
        and y_center + height / 2 <= 1
    )
