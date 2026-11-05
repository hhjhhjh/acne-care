from __future__ import annotations

import argparse
from pathlib import Path

from acne_care.dataset.external_label_editor import prepare_external_label_editor


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all external images for local bounding-box editing")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("runs/dataset-audit/external-full-editor"))
    args = parser.parse_args()
    document = prepare_external_label_editor(args.source_root, args.output_directory)
    print(f"images={document['summary']['images']}")
    print(f"annotations={document['summary']['annotations']}")
    print(f"editor={(args.output_directory / 'edit.html').resolve()}")


if __name__ == "__main__":
    main()
