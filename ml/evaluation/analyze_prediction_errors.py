from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from acne_care.evaluation.single_model_error_analysis import (
    analyze_prediction_document,
    rank_error_images,
    summarize_condition_groups,
)


def main() -> None:
    arguments = parse_arguments()
    document = json.loads(arguments.predictions.read_text(encoding="utf-8"))
    document["model_id"] = arguments.model_id
    analysis = analyze_prediction_document(
        document,
        confidence_threshold=arguments.confidence_threshold,
        iou_threshold=arguments.iou_threshold,
    )
    add_image_characteristics(
        analysis["per_image"],
        dataset_root=arguments.dataset_root,
        split=str(analysis["split"]),
    )
    analysis["by_brightness"] = summarize_condition_groups(
        analysis["per_image"], "brightness"
    )
    analysis["by_sharpness"] = summarize_condition_groups(
        analysis["per_image"], "sharpness"
    )
    analysis["by_density"] = summarize_condition_groups(
        analysis["per_image"], "ground_truth_count"
    )
    analysis["ranking"] = {
        "most_false_negatives": rank_error_images(
            analysis["per_image"], "false_negatives"
        ),
        "most_false_positives": rank_error_images(
            analysis["per_image"], "false_positives"
        ),
    }
    analysis["rules"] = {
        "size": "COCO area: small < 32^2, medium < 96^2, large otherwise",
        "condition_groups": "images divided into three equal-sized rank groups",
        "brightness": "mean grayscale intensity divided by 255",
        "sharpness": "variance of the grayscale Laplacian",
        "density": "ground-truth box count per image",
    }
    serialized = json.dumps(analysis, ensure_ascii=False, indent=2)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze false positives and false negatives for one model"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def add_image_characteristics(
    per_image: list[dict[str, Any]], *, dataset_root: Path, split: str
) -> None:
    source_split = {"val": "valid", "test": "test", "train": "train"}.get(split)
    if source_split is None:
        raise ValueError(f"unsupported dataset split: {split}")
    for item in per_image:
        image_path = dataset_root / source_split / "images" / Path(
            item["image_path"]
        ).name
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        item["image_path"] = image_path.relative_to(dataset_root).as_posix()
        item["brightness"] = float(gray.mean() / 255.0)
        item["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())


if __name__ == "__main__":
    main()
