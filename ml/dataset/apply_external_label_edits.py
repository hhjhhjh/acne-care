from __future__ import annotations

import argparse
import json
from pathlib import Path

from acne_care.dataset.external_label_edits import apply_completed_external_edits


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply completed external label edits and retain unreviewed labels")
    parser.add_argument("--editor-root", type=Path, required=True)
    parser.add_argument("--label-edits", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = apply_completed_external_edits(args.editor_root, args.label_edits, args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
