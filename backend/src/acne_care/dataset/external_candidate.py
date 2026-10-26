from __future__ import annotations

import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SPLITS = ("train", "valid", "test")
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ExternalDatasetAuditReport:
    root: Path
    image_counts: dict[str, int]
    annotated_image_counts: dict[str, int]
    annotation_counts: dict[str, int]
    class_counts: dict[str, int]
    unannotated_images: tuple[str, ...]
    cross_split_source_collisions: tuple[tuple[str, tuple[str, ...]], ...]
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class ImageMatch:
    candidate_path: str
    reference_path: str
    hamming_distance: int


@dataclass(frozen=True, slots=True)
class ReferenceMatchReport:
    exact_matches: tuple[ImageMatch, ...]
    near_matches: tuple[ImageMatch, ...]


@dataclass(frozen=True, slots=True)
class ReviewSampleResult:
    selected_images: int
    pimple_group_images: int
    acne_only_group_images: int
    annotations: int


def audit_roboflow_csv_dataset(root: Path) -> ExternalDatasetAuditReport:
    root = root.resolve()
    image_counts: dict[str, int] = {}
    annotated_image_counts: dict[str, int] = {}
    annotation_counts: dict[str, int] = {}
    class_counts: Counter[str] = Counter()
    unannotated_images: list[str] = []
    errors: list[str] = []
    source_splits: dict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        split_dir = root / split
        images = _list_images(split_dir)
        image_counts[split] = len(images)
        image_names = {path.name for path in images}
        for image_path in images:
            source_splits[_roboflow_source_name(image_path)].add(split)

        annotation_path = split_dir / "_annotations.csv"
        if not annotation_path.is_file():
            annotated_image_counts[split] = 0
            annotation_counts[split] = 0
            if images:
                errors.append(f"{split}/_annotations.csv is missing")
                unannotated_images.extend(
                    f"{split}/{image_name}" for image_name in sorted(image_names)
                )
            continue

        annotated_names, row_count = _audit_annotation_csv(
            root=root,
            split=split,
            annotation_path=annotation_path,
            image_names=image_names,
            class_counts=class_counts,
            errors=errors,
        )
        annotated_image_counts[split] = len(annotated_names)
        annotation_counts[split] = row_count
        unannotated_images.extend(
            f"{split}/{image_name}"
            for image_name in sorted(image_names - annotated_names)
        )

    collisions = tuple(
        (source_name, tuple(sorted(splits)))
        for source_name, splits in sorted(source_splits.items())
        if len(splits) > 1
    )
    return ExternalDatasetAuditReport(
        root=root,
        image_counts=image_counts,
        annotated_image_counts=annotated_image_counts,
        annotation_counts=annotation_counts,
        class_counts=dict(sorted(class_counts.items())),
        unannotated_images=tuple(sorted(unannotated_images)),
        cross_split_source_collisions=collisions,
        errors=tuple(sorted(errors)),
    )


def find_reference_image_matches(
    candidate_root: Path,
    reference_root: Path,
    max_hamming_distance: int = 6,
) -> ReferenceMatchReport:
    if max_hamming_distance < 0:
        raise ValueError("max_hamming_distance must be non-negative")

    candidate_root = candidate_root.resolve()
    reference_root = reference_root.resolve()
    candidate_images = _list_images_recursive(candidate_root)
    reference_images = _list_images_recursive(reference_root)

    references_by_sha: dict[str, list[Path]] = defaultdict(list)
    reference_fingerprints: list[tuple[Path, tuple[int, int]]] = []
    for image_path in reference_images:
        references_by_sha[_file_sha256(image_path)].append(image_path)
        reference_fingerprints.append((image_path, _perceptual_fingerprint(image_path)))

    exact_matches: list[ImageMatch] = []
    near_matches: list[ImageMatch] = []
    for candidate_path in candidate_images:
        candidate_sha = _file_sha256(candidate_path)
        exact_references = references_by_sha.get(candidate_sha, [])
        if exact_references:
            for reference_path in exact_references:
                exact_matches.append(
                    _image_match(
                        candidate_root,
                        candidate_path,
                        reference_root,
                        reference_path,
                        hamming_distance=0,
                    )
                )
            continue

        candidate_dhash, candidate_ahash = _perceptual_fingerprint(candidate_path)
        best_match: tuple[int, Path] | None = None
        for reference_path, (reference_dhash, reference_ahash) in reference_fingerprints:
            dhash_distance = (candidate_dhash ^ reference_dhash).bit_count()
            ahash_distance = (candidate_ahash ^ reference_ahash).bit_count()
            distance = max(dhash_distance, ahash_distance)
            if distance > max_hamming_distance:
                continue
            if best_match is None or distance < best_match[0]:
                best_match = (distance, reference_path)

        if best_match is not None:
            near_matches.append(
                _image_match(
                    candidate_root,
                    candidate_path,
                    reference_root,
                    best_match[1],
                    hamming_distance=best_match[0],
                )
            )

    return ReferenceMatchReport(
        exact_matches=tuple(sorted(exact_matches, key=_match_sort_key)),
        near_matches=tuple(sorted(near_matches, key=_match_sort_key)),
    )


def build_acne_pimple_review_sample(
    source_root: Path,
    output_root: Path,
    sample_size: int = 50,
    seed: int = 42,
) -> ReviewSampleResult:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("output_root must not be source_root or one of its parents")

    annotations_by_image: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for split in SPLITS:
        annotation_path = source_root / split / "_annotations.csv"
        if not annotation_path.is_file():
            continue
        with annotation_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                filename = (row.get("filename") or "").strip()
                if filename:
                    annotations_by_image[(split, filename)].append(row)

    pimple_group: list[tuple[str, str]] = []
    acne_only_group: list[tuple[str, str]] = []
    for key, rows in sorted(annotations_by_image.items()):
        classes = {(row.get("class") or "").strip() for row in rows}
        if "pimple" in classes:
            pimple_group.append(key)
        elif "acne" in classes:
            acne_only_group.append(key)

    eligible_count = len(pimple_group) + len(acne_only_group)
    if eligible_count < sample_size:
        raise ValueError(
            f"sample_size {sample_size} exceeds eligible image count {eligible_count}"
        )

    randomizer = random.Random(seed)
    pimple_target = min(sample_size // 2, len(pimple_group))
    acne_target = min(sample_size - pimple_target, len(acne_only_group))
    remaining = sample_size - pimple_target - acne_target
    if remaining:
        pimple_target += min(remaining, len(pimple_group) - pimple_target)
        remaining = sample_size - pimple_target - acne_target
    if remaining:
        acne_target += min(remaining, len(acne_only_group) - acne_target)

    selected_pimple = randomizer.sample(pimple_group, pimple_target)
    selected_acne = randomizer.sample(acne_only_group, acne_target)
    selected = sorted(
        [(key, "pimple") for key in selected_pimple]
        + [(key, "acne_only") for key in selected_acne]
    )

    temporary_root = output_root.with_name(f".{output_root.name}.tmp")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    for split in SPLITS:
        (temporary_root / split / "images").mkdir(parents=True, exist_ok=True)
        (temporary_root / split / "labels").mkdir(parents=True, exist_ok=True)

    manifest_items: list[dict[str, object]] = []
    annotation_count = 0
    for (source_split, filename), group in selected:
        source_image = source_root / source_split / filename
        if not source_image.is_file():
            raise FileNotFoundError(f"selected image is missing: {source_image}")
        output_name = f"{source_split}--{filename}"
        destination_image = temporary_root / "valid" / "images" / output_name
        shutil.copy2(source_image, destination_image)

        label_lines: list[str] = []
        included_classes: Counter[str] = Counter()
        for row in annotations_by_image[(source_split, filename)]:
            class_name = (row.get("class") or "").strip()
            if class_name not in {"acne", "pimple"}:
                continue
            width, height, xmin, ymin, xmax, ymax = (
                int(row[field])
                for field in ("width", "height", "xmin", "ymin", "xmax", "ymax")
            )
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            label_lines.append(
                "0 "
                f"{x_center:.10f} {y_center:.10f} "
                f"{box_width:.10f} {box_height:.10f}"
            )
            included_classes[class_name] += 1

        destination_label = (
            temporary_root / "valid" / "labels" / f"{Path(output_name).stem}.txt"
        )
        destination_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        annotation_count += len(label_lines)
        manifest_items.append(
            {
                "source_split": source_split,
                "source_filename": filename,
                "review_group": group,
                "included_class_counts": dict(sorted(included_classes.items())),
                "output_image": destination_image.relative_to(temporary_root).as_posix(),
            }
        )

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "sample_size": sample_size,
        "class_mapping": {"acne": 0, "pimple": 0},
        "excluded_classes": ["spot"],
        "items": manifest_items,
    }
    (temporary_root / "sample-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    temporary_root.replace(output_root)

    return ReviewSampleResult(
        selected_images=sample_size,
        pimple_group_images=pimple_target,
        acne_only_group_images=acne_target,
        annotations=annotation_count,
    )


def _audit_annotation_csv(
    root: Path,
    split: str,
    annotation_path: Path,
    image_names: set[str],
    class_counts: Counter[str],
    errors: list[str],
) -> tuple[set[str], int]:
    required_fields = {
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    }
    annotated_names: set[str] = set()
    row_count = 0
    with annotation_path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fields = set(reader.fieldnames or ())
        missing_fields = required_fields - fields
        if missing_fields:
            errors.append(
                f"{split}/_annotations.csv is missing columns: {', '.join(sorted(missing_fields))}"
            )
            return annotated_names, row_count

        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            filename = (row.get("filename") or "").strip()
            class_name = (row.get("class") or "").strip()
            if not filename:
                errors.append(f"{split}/_annotations.csv:{line_number} filename is empty")
                continue
            annotated_names.add(filename)
            if filename not in image_names:
                errors.append(f"{split}/{filename} is referenced but missing")
            if not class_name:
                errors.append(f"{split}/_annotations.csv:{line_number} class is empty")
            else:
                class_counts[class_name] += 1

            numeric = _parse_box(row, split, line_number, errors)
            if numeric is None:
                continue
            width, height, xmin, ymin, xmax, ymax = numeric
            if width <= 0 or height <= 0:
                errors.append(
                    f"{split}/_annotations.csv:{line_number} width and height must be positive"
                )
            if xmin >= xmax:
                errors.append(
                    f"{split}/_annotations.csv:{line_number} xmin must be less than xmax"
                )
            if ymin >= ymax:
                errors.append(
                    f"{split}/_annotations.csv:{line_number} ymin must be less than ymax"
                )
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                errors.append(
                    f"{split}/_annotations.csv:{line_number} box exceeds image bounds"
                )

    return annotated_names, row_count


def _parse_box(
    row: dict[str, str],
    split: str,
    line_number: int,
    errors: list[str],
) -> tuple[int, int, int, int, int, int] | None:
    fields = ("width", "height", "xmin", "ymin", "xmax", "ymax")
    try:
        return tuple(int(row[field]) for field in fields)  # type: ignore[return-value]
    except (TypeError, ValueError):
        errors.append(
            f"{split}/_annotations.csv:{line_number} box fields must be integers"
        )
        return None


def _roboflow_source_name(path: Path) -> str:
    return ROBOFLOW_SUFFIX.sub("", path.stem)


def _list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _list_images_recursive(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _perceptual_fingerprint(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as source_image:
            gray = source_image.convert("L")
            dhash_pixels = _flattened_pixels(
                gray.resize((9, 8), Image.Resampling.LANCZOS)
            )
            ahash_pixels = _flattened_pixels(
                gray.resize((8, 8), Image.Resampling.LANCZOS)
            )
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"cannot decode image: {path}") from error

    dhash = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            dhash = (dhash << 1) | int(
                dhash_pixels[offset + column] > dhash_pixels[offset + column + 1]
            )

    average = sum(ahash_pixels) / len(ahash_pixels)
    ahash = 0
    for pixel in ahash_pixels:
        ahash = (ahash << 1) | int(pixel >= average)
    return dhash, ahash


def _flattened_pixels(image: Image.Image) -> list[int]:
    if hasattr(image, "get_flattened_data"):
        return list(image.get_flattened_data())
    return list(image.getdata())


def _image_match(
    candidate_root: Path,
    candidate_path: Path,
    reference_root: Path,
    reference_path: Path,
    hamming_distance: int,
) -> ImageMatch:
    return ImageMatch(
        candidate_path=candidate_path.relative_to(candidate_root).as_posix(),
        reference_path=reference_path.relative_to(reference_root).as_posix(),
        hamming_distance=hamming_distance,
    )


def _match_sort_key(match: ImageMatch) -> tuple[str, str, int]:
    return match.candidate_path, match.reference_path, match.hamming_distance
