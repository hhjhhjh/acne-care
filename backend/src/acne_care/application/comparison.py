from __future__ import annotations

from datetime import date
from typing import Any


INFERENCE_SETTINGS = ("model_id", "weights_sha256", "confidence_threshold", "image_size", "nms_iou", "device")


def compare_analyses(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Compare persisted counts; never infer skin improvement or rerun a detector."""
    if previous["id"] == current["id"]:
        raise ValueError("서로 다른 기록 두 개를 선택해 주세요.")
    days_between = (date.fromisoformat(current["captured_on"]) - date.fromisoformat(previous["captured_on"])).days
    if days_between < 0:
        raise ValueError("현재 기록의 촬영일은 이전 기록의 촬영일과 같거나 이후여야 합니다.")
    differences = [
        {"field": field, "previous": previous[field], "current": current[field]}
        for field in INFERENCE_SETTINGS if previous[field] != current[field]
    ]
    return {
        "days_between": days_between,
        "count_delta": current["count"] - previous["count"],
        "analysis_settings_match": not differences,
        "setting_differences": differences,
    }
