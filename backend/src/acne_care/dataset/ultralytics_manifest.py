from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}


@dataclass(frozen=True, slots=True)
class UltralyticsManifest:
    dataset_yaml: Path
    image_counts: dict[str, int]


def build_ultralytics_manifest(
    source_root: Path,
    output_dir: Path,
    excluded_paths: set[str],
    class_names: tuple[str, ...],
) -> UltralyticsManifest:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    if not class_names:
        raise ValueError("at least one class name is required")

    split_images: dict[str, list[Path]] = {}
    all_relative_paths: set[str] = set()
    for manifest_split, source_split in SPLITS.items():
        image_dir = source_root / source_split / "images"
        label_dir = source_root / source_split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"missing images or labels directory for {source_split} split"
            )

        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in images:
            relative_path = image_path.relative_to(source_root).as_posix()
            all_relative_paths.add(relative_path)
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"missing label for {relative_path}")

        split_images[manifest_split] = [
            image_path
            for image_path in images
            if image_path.relative_to(source_root).as_posix() not in excluded_paths
        ]

    unknown_exclusions = excluded_paths - all_relative_paths
    if unknown_exclusions:
        raise ValueError(
            "unknown exclusion paths: " + ", ".join(sorted(unknown_exclusions))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, Path] = {}
    for split, images in split_images.items():
        if not images:
            raise ValueError(f"no images remain in {split} split")
        manifest_path = output_dir / f"{split}.txt"
        manifest_path.write_text(
            "".join(f"{image_path.resolve().as_posix()}\n" for image_path in images),
            encoding="utf-8",
        )
        manifest_paths[split] = manifest_path

    dataset_yaml = output_dir / "data.yaml"
    yaml_lines = [
        f"train: {json.dumps(manifest_paths['train'].as_posix())}",
        f"val: {json.dumps(manifest_paths['val'].as_posix())}",
        f"test: {json.dumps(manifest_paths['test'].as_posix())}",
        "names:",
    ]
    yaml_lines.extend(
        f"  {class_id}: {json.dumps(class_name)}"
        for class_id, class_name in enumerate(class_names)
    )
    dataset_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return UltralyticsManifest(
        dataset_yaml=dataset_yaml,
        image_counts={split: len(images) for split, images in split_images.items()},
    )
