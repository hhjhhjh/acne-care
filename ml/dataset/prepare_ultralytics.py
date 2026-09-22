from __future__ import annotations

import argparse
from pathlib import Path

from acne_care.dataset import build_ultralytics_manifest


def main() -> None:
    arguments = parse_arguments()
    excluded_paths = read_exclusions(arguments.exclude_file)
    result = build_ultralytics_manifest(
        source_root=arguments.source_root,
        output_dir=arguments.output_dir,
        excluded_paths=excluded_paths,
        class_names=("Acne",),
    )
    print(f"dataset_yaml={result.dataset_yaml}")
    for split, count in result.image_counts.items():
        print(f"{split}_images={count}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local Ultralytics manifests without copying images"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-file", type=Path)
    return parser.parse_args()


def read_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }


if __name__ == "__main__":
    main()
