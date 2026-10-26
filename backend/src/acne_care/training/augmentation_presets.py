from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class TransformSpec:
    name: str
    arguments: dict[str, float | int]


@dataclass(frozen=True)
class AugmentationSettings:
    name: str
    yolo_overrides: dict[str, float | int]
    custom_transforms: tuple[TransformSpec, ...]


_CONTROL_OVERRIDES: dict[str, float | int] = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "close_mosaic": 10,
}

_PRESET_NAMES = ("control", "contrast", "rotation", "safe-crop")


def available_augmentation_presets() -> tuple[str, ...]:
    return _PRESET_NAMES


def build_augmentation_settings(name: str, image_size: int) -> AugmentationSettings:
    if image_size <= 0:
        raise ValueError("image size must be positive")
    if name not in _PRESET_NAMES:
        raise ValueError(f"unknown augmentation preset: {name}")

    yolo_overrides = dict(_CONTROL_OVERRIDES)
    custom_transforms: tuple[TransformSpec, ...] = ()

    if name == "contrast":
        custom_transforms = (
            TransformSpec(
                name="RandomBrightnessContrast",
                arguments={
                    "brightness_limit": 0.0,
                    "contrast_limit": 0.15,
                    "p": 0.5,
                },
            ),
        )
    elif name == "rotation":
        yolo_overrides["degrees"] = 10.0
    elif name == "safe-crop":
        custom_transforms = (
            TransformSpec(
                name="RandomSizedBBoxSafeCrop",
                arguments={
                    "height": image_size,
                    "width": image_size,
                    "erosion_rate": 0.0,
                    "p": 0.3,
                },
            ),
        )

    return AugmentationSettings(
        name=name,
        yolo_overrides=yolo_overrides,
        custom_transforms=custom_transforms,
    )


def instantiate_custom_transforms(
    settings: AugmentationSettings,
    albumentations_module: ModuleType | Any | None = None,
) -> list[Any]:
    if not settings.custom_transforms:
        return []

    if albumentations_module is None:
        try:
            import albumentations as albumentations_module
        except ImportError as error:
            raise RuntimeError(
                "the selected augmentation preset requires albumentations"
            ) from error

    transforms = []
    for specification in settings.custom_transforms:
        try:
            transform_class = getattr(albumentations_module, specification.name)
        except AttributeError as error:
            raise RuntimeError(
                f"albumentations does not provide {specification.name}"
            ) from error
        transforms.append(transform_class(**specification.arguments))
    return transforms
