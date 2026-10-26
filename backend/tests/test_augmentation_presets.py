import unittest

from acne_care.training.augmentation_presets import (
    available_augmentation_presets,
    build_augmentation_settings,
)


class AugmentationPresetTest(unittest.TestCase):
    def test_exposes_only_the_first_controlled_experiment_batch(self) -> None:
        self.assertEqual(
            available_augmentation_presets(),
            ("control", "contrast", "rotation", "safe-crop"),
        )

    def test_control_keeps_the_recorded_yolo_defaults(self) -> None:
        settings = build_augmentation_settings("control", image_size=640)

        self.assertEqual(
            settings.yolo_overrides,
            {
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
            },
        )
        self.assertEqual(settings.custom_transforms, ())

    def test_contrast_changes_only_contrast_beyond_the_control(self) -> None:
        control = build_augmentation_settings("control", image_size=640)
        contrast = build_augmentation_settings("contrast", image_size=640)

        self.assertEqual(contrast.yolo_overrides, control.yolo_overrides)
        self.assertEqual(len(contrast.custom_transforms), 1)
        transform = contrast.custom_transforms[0]
        self.assertEqual(transform.name, "RandomBrightnessContrast")
        self.assertEqual(
            transform.arguments,
            {
                "brightness_limit": 0.0,
                "contrast_limit": 0.15,
                "p": 0.5,
            },
        )

    def test_rotation_changes_only_degrees_beyond_the_control(self) -> None:
        control = build_augmentation_settings("control", image_size=640)
        rotation = build_augmentation_settings("rotation", image_size=640)

        expected = dict(control.yolo_overrides)
        expected["degrees"] = 10.0
        self.assertEqual(rotation.yolo_overrides, expected)
        self.assertEqual(rotation.custom_transforms, ())

    def test_safe_crop_preserves_every_box_and_uses_training_size(self) -> None:
        crop = build_augmentation_settings("safe-crop", image_size=640)

        self.assertEqual(len(crop.custom_transforms), 1)
        transform = crop.custom_transforms[0]
        self.assertEqual(transform.name, "RandomSizedBBoxSafeCrop")
        self.assertEqual(
            transform.arguments,
            {
                "height": 640,
                "width": 640,
                "erosion_rate": 0.0,
                "p": 0.3,
            },
        )

    def test_rejects_unknown_preset(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown augmentation preset"):
            build_augmentation_settings("everything", image_size=640)

    def test_rejects_non_positive_image_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "image size must be positive"):
            build_augmentation_settings("control", image_size=0)


if __name__ == "__main__":
    unittest.main()
