from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SPLIT_ALIASES = {
    "train": ("train",),
    "valid": ("valid", "val"),
    "test": ("test",),
}


@dataclass(frozen=True, slots=True)
class DatasetAuditReport:
    root: Path
    image_counts: dict[str, int]
    label_counts: dict[str, int]
    class_ids: tuple[int, ...]
    cross_split_duplicates: tuple[tuple[str, str], ...]
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def audit_yolo_dataset(root: Path) -> DatasetAuditReport:
    root = root.resolve()
    image_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    class_ids: set[int] = set()
    errors: list[str] = []
    content_locations: dict[str, tuple[str, str]] = {}
    duplicates: list[tuple[str, str]] = []

    for canonical_split, aliases in SPLIT_ALIASES.items():
        split_name = _find_split(root, aliases)
        if split_name is None:
            image_counts[canonical_split] = 0
            label_counts[canonical_split] = 0
            errors.append(
                f"missing {canonical_split} split; expected one of: {', '.join(aliases)}"
            )
            continue

        image_dir = root / split_name / "images"
        label_dir = root / split_name / "labels"
        images = _list_images(image_dir)
        labels = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        image_counts[canonical_split] = len(images)
        label_counts[canonical_split] = len(labels)

        if not image_dir.is_dir():
            errors.append(f"{split_name}/images directory is missing")
        if not label_dir.is_dir():
            errors.append(f"{split_name}/labels directory is missing")

        image_stems = {image_path.stem for image_path in images}
        for image_path in images:
            relative_image = image_path.relative_to(root).as_posix()
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                errors.append(f"{relative_image} has no matching label")
            else:
                _audit_label(root, label_path, class_ids, errors)

            content_hash = _file_sha256(image_path)
            previous = content_locations.get(content_hash)
            if previous is not None and previous[0] != canonical_split:
                duplicates.append((relative_image, previous[1]))
                errors.append(
                    f"exact duplicate crosses splits: {relative_image} and {previous[1]}"
                )
            else:
                content_locations.setdefault(
                    content_hash,
                    (canonical_split, relative_image),
                )

        for label_path in labels:
            if label_path.stem not in image_stems:
                relative_label = label_path.relative_to(root).as_posix()
                errors.append(f"{relative_label} has no matching image")

    return DatasetAuditReport(
        root=root,
        image_counts=image_counts,
        label_counts=label_counts,
        class_ids=tuple(sorted(class_ids)),
        cross_split_duplicates=tuple(sorted(duplicates)),
        errors=tuple(sorted(errors)),
    )


def _find_split(root: Path, aliases: tuple[str, ...]) -> str | None:
    matches = [alias for alias in aliases if (root / alias).is_dir()]
    if not matches:
        return None
    return matches[0]


def _list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _audit_label(
    root: Path,
    label_path: Path,
    class_ids: set[int],
    errors: list[str],
) -> None:
    relative_label = label_path.relative_to(root).as_posix()
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        errors.append(f"{relative_label} is not valid UTF-8")
        return

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(
                f"{relative_label}:{line_number} must contain class_id, x, y, width, height"
            )
            continue

        try:
            class_value = float(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError:
            errors.append(f"{relative_label}:{line_number} contains a non-numeric value")
            continue

        if not class_value.is_integer() or class_value < 0:
            errors.append(
                f"{relative_label}:{line_number} class_id must be a non-negative integer"
            )
            continue

        class_id = int(class_value)
        class_ids.add(class_id)
        x_center, y_center, width, height = coordinates
        if not all(isfinite(value) for value in coordinates):
            errors.append(f"{relative_label}:{line_number} coordinates must be finite")
            continue
        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
            and 0 <= x_center - width / 2
            and x_center + width / 2 <= 1
            and 0 <= y_center - height / 2
            and y_center + height / 2 <= 1
        ):
            errors.append(
                f"{relative_label}:{line_number} normalized coordinates exceed image bounds"
            )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
