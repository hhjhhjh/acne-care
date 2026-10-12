from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from acne_care.dataset.label_review import build_label_review
from acne_care.dataset.label_review_dashboard import render_dashboard_html


REVIEW_SPLITS = ("valid", "test")


def main() -> None:
    arguments = parse_arguments()
    document = build_label_review(arguments.dataset_root)
    inspect_images(arguments.dataset_root.resolve(), document)
    refresh_hint_counts(document)

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    audit_path = arguments.output_directory / "audit.json"
    audit_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    copy_review_assets(
        arguments.dataset_root.resolve(),
        arguments.output_directory,
        document,
    )
    dashboard_path = arguments.output_directory / "review.html"
    dashboard_path.write_text(
        render_dashboard_html(document, splits=REVIEW_SPLITS),
        encoding="utf-8",
    )
    print_summary(document, audit_path, dashboard_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a YOLO dataset and build a local label-review dashboard"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("runs/dataset-audit"),
    )
    return parser.parse_args()


def inspect_images(root: Path, document: dict[str, Any]) -> None:
    for item in document["items"]:
        image_path = root / item["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            item["review_hints"].append("unreadable_image")
            item["details"].append("the image could not be decoded")
            continue
        height, width = image.shape[:2]
        item["image_width"] = width
        item["image_height"] = height
        item["review_hints"] = sorted(set(item["review_hints"]))


def refresh_hint_counts(document: dict[str, Any]) -> None:
    counts = Counter(
        hint for item in document["items"] for hint in item["review_hints"]
    )
    document["summary"]["review_hint_counts"] = dict(sorted(counts.items()))


def copy_review_assets(
    root: Path,
    output_directory: Path,
    document: dict[str, Any],
) -> None:
    for item in document["items"]:
        if item["split"] not in REVIEW_SPLITS:
            continue
        source = root / item["image_path"]
        destination = (
            output_directory
            / "assets"
            / item["split"]
            / Path(item["image_path"]).name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or source.stat().st_mtime_ns > destination.stat().st_mtime_ns:
            shutil.copy2(source, destination)


def print_summary(
    document: dict[str, Any],
    audit_path: Path,
    dashboard_path: Path,
) -> None:
    summary = document["summary"]
    print(f"images: {summary['images']}")
    print(f"annotations: {summary['annotations']}")
    print(f"structural errors: {summary['structural_error_count']}")
    print(f"review hints: {summary['review_hint_counts']}")
    print(f"audit: {audit_path.resolve()}")
    print(f"dashboard: {dashboard_path.resolve()}")


if __name__ == "__main__":
    main()
