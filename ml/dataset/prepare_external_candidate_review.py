from __future__ import annotations

import argparse
from pathlib import Path

from acne_care.dataset.external_candidate import build_acne_pimple_review_sample


def main() -> None:
    arguments = parse_arguments()
    result = build_acne_pimple_review_sample(
        source_root=arguments.source_root,
        output_root=arguments.output_root,
        sample_size=arguments.sample_size,
        seed=arguments.seed,
    )
    print(f"output={arguments.output_root}")
    print(f"selected_images={result.selected_images}")
    print(f"pimple_group_images={result.pimple_group_images}")
    print(f"acne_only_group_images={result.acne_only_group_images}")
    print(f"annotations={result.annotations}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic YOLO review sample by mapping candidate acne "
            "and pimple boxes to one Acne class"
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
