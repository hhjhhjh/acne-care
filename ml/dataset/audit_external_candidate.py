from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from acne_care.dataset.external_candidate import (
    audit_roboflow_csv_dataset,
    find_reference_image_matches,
)


def main() -> None:
    arguments = parse_arguments()
    dataset_report = audit_roboflow_csv_dataset(arguments.candidate_root)
    match_report = find_reference_image_matches(
        candidate_root=arguments.candidate_root,
        reference_root=arguments.reference_root,
        max_hamming_distance=arguments.max_hamming_distance,
    )
    result = {
        "source": {
            "kaggle_ref": arguments.kaggle_ref,
            "declared_license": arguments.declared_license,
        },
        "candidate": {
            "image_counts": dataset_report.image_counts,
            "annotated_image_counts": dataset_report.annotated_image_counts,
            "annotation_counts": dataset_report.annotation_counts,
            "class_counts": dataset_report.class_counts,
            "unannotated_image_count": len(dataset_report.unannotated_images),
            "cross_split_source_collision_count": len(
                dataset_report.cross_split_source_collisions
            ),
            "cross_split_source_collisions": [
                {"source_name": source_name, "splits": list(splits)}
                for source_name, splits in dataset_report.cross_split_source_collisions
            ],
            "errors": list(dataset_report.errors),
        },
        "reference_comparison": {
            "max_hamming_distance": arguments.max_hamming_distance,
            "exact_match_count": len(match_report.exact_matches),
            "near_match_count": len(match_report.near_matches),
            "exact_matches": [asdict(match) for match in match_report.exact_matches],
            "near_matches": [asdict(match) for match in match_report.near_matches],
        },
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output={arguments.output}")
    print(f"images={sum(dataset_report.image_counts.values())}")
    print(f"annotations={sum(dataset_report.annotation_counts.values())}")
    print(f"classes={dataset_report.class_counts}")
    print(f"unannotated_images={len(dataset_report.unannotated_images)}")
    print(
        "cross_split_source_collisions="
        f"{len(dataset_report.cross_split_source_collisions)}"
    )
    print(f"structural_errors={len(dataset_report.errors)}")
    print(f"exact_reference_matches={len(match_report.exact_matches)}")
    print(f"near_reference_matches={len(match_report.near_matches)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a Roboflow CSV object-detection candidate and compare its images "
            "with an existing reference dataset"
        )
    )
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kaggle-ref", required=True)
    parser.add_argument("--declared-license", required=True)
    parser.add_argument("--max-hamming-distance", type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    main()
