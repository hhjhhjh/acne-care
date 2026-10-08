from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from acne_care.dataset.label_editor_dashboard import (
    ACTIONABLE_STATUSES,
    render_label_editor_html,
)
from acne_care.dataset.label_review_dashboard import review_dataset_fingerprint


def main() -> None:
    arguments = parse_arguments()
    audit_document = read_json(arguments.audit_json)
    review_results = read_json(arguments.review_results)
    expected_fingerprint = review_dataset_fingerprint(audit_document)
    if review_results.get("dataset_fingerprint") != expected_fingerprint:
        raise ValueError("review results belong to a different dataset version")

    actionable_paths = {
        path
        for path, review in review_results["reviews"].items()
        if review.get("status") in ACTIONABLE_STATUSES
    }
    copy_assets(
        audit_document,
        actionable_paths=actionable_paths,
        output_directory=arguments.output_directory,
    )
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    editor_path = arguments.output_directory / "edit.html"
    editor_path.write_text(
        render_label_editor_html(audit_document, review_results),
        encoding="utf-8",
    )
    print(f"actionable images: {len(actionable_paths)}")
    print(f"editor: {editor_path.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a non-destructive YOLO label editor from review results"
    )
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--review-results", type=Path, required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("runs/dataset-audit"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_assets(
    audit_document: dict[str, Any],
    *,
    actionable_paths: set[str],
    output_directory: Path,
) -> None:
    dataset_root = Path(audit_document["dataset_root"])
    for item in audit_document["items"]:
        if item["image_path"] not in actionable_paths:
            continue
        source = dataset_root / item["image_path"]
        destination = (
            output_directory
            / "assets"
            / item["split"]
            / Path(item["image_path"]).name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or source.stat().st_mtime_ns > destination.stat().st_mtime_ns:
            shutil.copy2(source, destination)


if __name__ == "__main__":
    main()
