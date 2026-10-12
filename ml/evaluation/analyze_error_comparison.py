from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from acne_care.evaluation.error_analysis import compare_prediction_documents


def main() -> None:
    arguments = parse_arguments()
    legacy = read_json(arguments.legacy_predictions)
    yolo11 = read_json(arguments.yolo11_predictions)
    comparison = compare_prediction_documents(legacy, yolo11)
    add_image_characteristics(comparison["per_image"])
    comparison["by_brightness"] = condition_summary(
        comparison["per_image"], "brightness"
    )
    comparison["by_sharpness"] = condition_summary(
        comparison["per_image"], "sharpness"
    )
    comparison["ranking"] = {
        "most_yolo11_false_negatives": ranked_images(
            comparison["per_image"], metric="false_negatives", model="yolo11"
        ),
        "most_yolo11_false_positives": ranked_images(
            comparison["per_image"], metric="false_positives", model="yolo11"
        ),
    }
    comparison["metadata"] = {
        "legacy_model_id": legacy["model_id"],
        "legacy_confidence_threshold": legacy["confidence_threshold"],
        "yolo11_model_id": yolo11["model_id"],
        "yolo11_confidence_threshold": yolo11["confidence_threshold"],
        "iou_threshold": legacy["iou_threshold"],
        "split": legacy["split"],
        "size_rule": "COCO area: small < 32^2, medium < 96^2, large otherwise",
        "condition_rule": "images divided into three equal-sized rank groups",
    }
    normalize_image_paths(comparison)
    serialized = json.dumps(comparison, ensure_ascii=False, indent=2)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy and YOLO11 prediction errors"
    )
    parser.add_argument("--legacy-predictions", type=Path, required=True)
    parser.add_argument("--yolo11-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_image_characteristics(per_image: list[dict[str, Any]]) -> None:
    for item in per_image:
        image = cv2.imread(item["image_path"])
        if image is None:
            raise ValueError(f"failed to read image: {item['image_path']}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        item["brightness"] = float(gray.mean() / 255.0)
        item["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())


def condition_summary(
    per_image: list[dict[str, Any]], metric: str
) -> dict[str, dict[str, int | float]]:
    ordered = sorted(per_image, key=lambda item: item[metric])
    groups = {"low": [], "middle": [], "high": []}
    names = tuple(groups)
    for index, item in enumerate(ordered):
        group_index = min(index * 3 // len(ordered), 2)
        groups[names[group_index]].append(item)

    return {
        name: summarize_images(items, metric)
        for name, items in groups.items()
    }


def summarize_images(
    items: list[dict[str, Any]], metric: str
) -> dict[str, int | float]:
    ground_truths = sum(item["ground_truth_count"] for item in items)
    legacy_detected = sum(
        item["outcomes"]["both_detected"] + item["outcomes"]["legacy_only"]
        for item in items
    )
    yolo11_detected = sum(
        item["outcomes"]["both_detected"] + item["outcomes"]["yolo11_only"]
        for item in items
    )
    return {
        "images": len(items),
        "ground_truths": ground_truths,
        f"{metric}_minimum": min(item[metric] for item in items),
        f"{metric}_maximum": max(item[metric] for item in items),
        "legacy_recall": legacy_detected / ground_truths if ground_truths else 0.0,
        "yolo11_recall": yolo11_detected / ground_truths if ground_truths else 0.0,
        "legacy_false_positives": sum(
            item["false_positives"]["legacy"] for item in items
        ),
        "yolo11_false_positives": sum(
            item["false_positives"]["yolo11"] for item in items
        ),
    }


def ranked_images(
    per_image: list[dict[str, Any]], *, metric: str, model: str, limit: int = 10
) -> list[dict[str, Any]]:
    if metric == "false_negatives":
        score = lambda item: (
            item["outcomes"]["both_missed"] + item["outcomes"]["legacy_only"]
        )
    elif metric == "false_positives":
        score = lambda item: item["false_positives"][model]
    else:
        raise ValueError(f"unsupported ranking metric: {metric}")
    return [
        {
            "image_path": item["image_path"],
            "count": score(item),
            "ground_truth_count": item["ground_truth_count"],
            "brightness": item["brightness"],
            "sharpness": item["sharpness"],
        }
        for item in sorted(per_image, key=score, reverse=True)[:limit]
    ]


def normalize_image_paths(comparison: dict[str, Any]) -> None:
    for item in comparison["per_image"]:
        item["image_path"] = portable_image_path(item["image_path"])
    for ranking in comparison["ranking"].values():
        for item in ranking:
            item["image_path"] = portable_image_path(item["image_path"])


def portable_image_path(path: str) -> str:
    parts = Path(path).parts
    if "valid" in parts:
        index = parts.index("valid")
        return Path(*parts[index:]).as_posix()
    return Path(path).name


if __name__ == "__main__":
    main()
