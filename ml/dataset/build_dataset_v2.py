from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from acne_care.dataset.label_edits import build_dataset_v2


def main() -> None:
    arguments = parse_arguments()
    result = build_dataset_v2(
        arguments.source_root,
        arguments.output_root,
        _read_json(arguments.audit_json),
        _read_json(arguments.label_edits),
        preexisting_exclusions=_read_exclusions(arguments.exclude_file),
    )
    print(f"dataset-v2: {result.root}")
    print(f"images: {result.image_counts}")
    print(f"annotations: {result.annotation_counts}")
    print(f"edited images: {result.edited_images}")
    print(f"excluded images: {result.excluded_images}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an independent YOLO dataset from completed label edits"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--label-edits", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exclude-file", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }


if __name__ == "__main__":
    main()
