from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from acne_care.dataset.audit import IMAGE_SUFFIXES, SPLIT_ALIASES
from acne_care.dataset.label_review_dashboard import review_dataset_fingerprint


@dataclass(frozen=True, slots=True)
class DatasetV2Build:
    root: Path
    image_counts: dict[str, int]
    annotation_counts: dict[str, int]
    edited_images: int
    excluded_images: int


def build_dataset_v2(
    source_root: Path,
    output_root: Path,
    audit_document: dict[str, Any],
    edit_document: dict[str, Any],
    *,
    preexisting_exclusions: set[str] | None = None,
) -> DatasetV2Build:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    edits = _validate_inputs(source_root, output_root, audit_document, edit_document)
    preexisting_exclusions = preexisting_exclusions or set()
    _validate_exclusions(source_root, edits, preexisting_exclusions)
    temporary_root = output_root.with_name(
        f".{output_root.name}.building-{uuid4().hex}"
    )

    try:
        _copy_dataset(source_root, temporary_root)
        _apply_edits(temporary_root, edits)
        _apply_exclusions(temporary_root, preexisting_exclusions)
        result = _summarize(
            temporary_root,
            edits,
            excluded_images=sum(bool(edit.get("excluded")) for edit in edits.values())
            + len(preexisting_exclusions),
        )
        _write_dataset_yaml(temporary_root, dataset_root=output_root)
        _write_build_summary(temporary_root, result, edit_document)
        if output_root.exists():
            shutil.rmtree(output_root)
        temporary_root.rename(output_root)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise

    return DatasetV2Build(
        root=output_root,
        image_counts=result.image_counts,
        annotation_counts=result.annotation_counts,
        edited_images=result.edited_images,
        excluded_images=result.excluded_images,
    )


def _validate_inputs(
    source_root: Path,
    output_root: Path,
    audit_document: dict[str, Any],
    edit_document: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if source_root == output_root:
        raise ValueError("output_root must differ from source_root")
    if not source_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {source_root}")
    if edit_document.get("schema_version") != 1:
        raise ValueError("unsupported label edit schema version")
    expected_fingerprint = review_dataset_fingerprint(audit_document)
    if edit_document.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError("label edits belong to a different dataset version")
    edits = edit_document.get("edits")
    if not isinstance(edits, dict):
        raise ValueError("label edits must contain an edits object")

    audit_paths = {str(item["image_path"]) for item in audit_document["items"]}
    unknown_paths = set(edits) - audit_paths
    if unknown_paths:
        raise ValueError("unknown edited image: " + ", ".join(sorted(unknown_paths)))
    incomplete = [path for path, edit in edits.items() if not edit.get("completed")]
    if incomplete:
        raise ValueError("label edits are not completed: " + ", ".join(incomplete))

    for image_path, edit in edits.items():
        source_image = source_root / image_path
        source_label = _label_path(source_root, image_path)
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(f"edited sample is missing: {image_path}")
        annotations = edit.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"annotations must be a list: {image_path}")
        for annotation in annotations:
            _format_annotation(annotation)
    return edits


def _validate_exclusions(
    source_root: Path,
    edits: dict[str, dict[str, Any]],
    excluded_paths: set[str],
) -> None:
    overlap = excluded_paths & set(edits)
    if overlap:
        raise ValueError(
            "preexisting exclusions overlap label edits: "
            + ", ".join(sorted(overlap))
        )
    unknown = [path for path in excluded_paths if not (source_root / path).is_file()]
    if unknown:
        raise ValueError("unknown preexisting exclusion: " + ", ".join(sorted(unknown)))
    for image_path in excluded_paths:
        if not _label_path(source_root, image_path).is_file():
            raise FileNotFoundError(f"excluded sample label is missing: {image_path}")


def _copy_dataset(source_root: Path, destination_root: Path) -> None:
    destination_root.mkdir(parents=True)
    for aliases in SPLIT_ALIASES.values():
        split_name = next((name for name in aliases if (source_root / name).is_dir()), None)
        if split_name is None:
            raise FileNotFoundError(f"dataset split is missing: {aliases[0]}")
        shutil.copytree(source_root / split_name, destination_root / split_name)


def _apply_edits(
    dataset_root: Path,
    edits: dict[str, dict[str, Any]],
) -> None:
    for image_path, edit in edits.items():
        target_image = dataset_root / image_path
        target_label = _label_path(dataset_root, image_path)
        if edit.get("excluded"):
            target_image.unlink()
            target_label.unlink()
            continue
        target_label.write_text(
            "".join(_format_annotation(item) for item in edit["annotations"]),
            encoding="utf-8",
        )


def _apply_exclusions(dataset_root: Path, excluded_paths: set[str]) -> None:
    for image_path in excluded_paths:
        (dataset_root / image_path).unlink()
        _label_path(dataset_root, image_path).unlink()


def _format_annotation(annotation: dict[str, Any]) -> str:
    required_fields = ("class_id", "x_center", "y_center", "width", "height")
    if any(field not in annotation for field in required_fields):
        raise ValueError("annotation is missing a required field")
    class_id = annotation["class_id"]
    if not isinstance(class_id, int) or class_id < 0:
        raise ValueError("annotation class_id must be a non-negative integer")
    values = [float(annotation[field]) for field in required_fields[1:]]
    x_center, y_center, width, height = values
    if not (
        0 <= x_center <= 1
        and 0 <= y_center <= 1
        and 0 < width <= 1
        and 0 < height <= 1
        and x_center - width / 2 >= 0
        and x_center + width / 2 <= 1
        and y_center - height / 2 >= 0
        and y_center + height / 2 <= 1
    ):
        raise ValueError("annotation coordinates must describe a box inside the image")
    coordinates = " ".join(str(value) for value in values)
    return f"{class_id} {coordinates}\n"


def _label_path(root: Path, image_path: str) -> Path:
    relative_image = Path(image_path)
    if len(relative_image.parts) != 3 or relative_image.parts[1] != "images":
        raise ValueError(f"unexpected dataset image path: {image_path}")
    return root / relative_image.parts[0] / "labels" / f"{relative_image.stem}.txt"


def _summarize(
    root: Path,
    edits: dict[str, dict[str, Any]],
    *,
    excluded_images: int,
) -> DatasetV2Build:
    image_counts: dict[str, int] = {}
    annotation_counts: dict[str, int] = {}
    for canonical_split, aliases in SPLIT_ALIASES.items():
        split_name = next((name for name in aliases if (root / name).is_dir()), None)
        if split_name is None:
            raise FileNotFoundError(f"dataset split is missing: {canonical_split}")
        image_dir = root / split_name / "images"
        label_dir = root / split_name / "labels"
        images = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        labels = list(label_dir.glob("*.txt"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"image and label files do not match in {split_name}")
        image_counts[canonical_split] = len(images)
        annotation_counts[canonical_split] = sum(
            len(path.read_text(encoding="utf-8").splitlines()) for path in labels
        )
    return DatasetV2Build(
        root=root,
        image_counts=image_counts,
        annotation_counts=annotation_counts,
        edited_images=sum(not edit.get("excluded") for edit in edits.values()),
        excluded_images=excluded_images,
    )


def _write_dataset_yaml(root: Path, *, dataset_root: Path) -> None:
    root.joinpath("data.yaml").write_text(
        "\n".join(
            (
                f'path: "{dataset_root.resolve().as_posix()}"',
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "names:",
                '  0: "Acne"',
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_build_summary(
    root: Path,
    result: DatasetV2Build,
    edit_document: dict[str, Any],
) -> None:
    summary = {
        "schema_version": 1,
        "dataset_fingerprint": edit_document["dataset_fingerprint"],
        "image_counts": result.image_counts,
        "annotation_counts": result.annotation_counts,
        "edited_images": result.edited_images,
        "excluded_images": result.excluded_images,
    }
    root.joinpath("build-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
