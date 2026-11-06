import unittest

from acne_care.application.comparison import compare_analyses


def record(analysis_id, captured_on, count, **changes):
    return {
        "id": analysis_id, "captured_on": captured_on, "count": count,
        "model_id": "control", "weights_sha256": "a" * 64,
        "confidence_threshold": 0.3, "image_size": 640,
        "nms_iou": 0.7, "device": "cpu", **changes,
    }


class AnalysisComparisonTest(unittest.TestCase):
    def test_counts_are_current_minus_previous_and_dates_use_capture_day(self):
        previous = record("older", "2026-10-01", 8)
        current = record("newer", "2026-10-06", 5)
        result = compare_analyses(previous, current)
        self.assertEqual(result["count_delta"], -3)
        self.assertEqual(result["days_between"], 5)
        self.assertTrue(result["analysis_settings_match"])
        self.assertEqual(result["setting_differences"], [])

    def test_zero_counts_increases_and_same_day_distinct_records(self):
        for before, after, delta in [(0, 0, 0), (0, 3, 3), (3, 0, -3)]:
            with self.subTest(before=before, after=after):
                result = compare_analyses(record("a", "2026-10-01", before),
                                          record("b", "2026-10-01", after))
                self.assertEqual(result["count_delta"], delta)
                self.assertEqual(result["days_between"], 0)

    def test_same_record_and_reversed_capture_dates_are_rejected(self):
        previous = record("a", "2026-10-06", 2)
        with self.assertRaisesRegex(ValueError, "서로 다른"):
            compare_analyses(previous, previous)
        with self.assertRaisesRegex(ValueError, "촬영일"):
            compare_analyses(previous, record("b", "2026-10-01", 4))

    def test_every_inference_setting_difference_is_reported(self):
        changes = {"model_id": "other", "weights_sha256": "b" * 64,
                   "confidence_threshold": 0.4, "image_size": 832,
                   "nms_iou": 0.6, "device": "cuda:0"}
        previous = record("a", "2026-10-01", 5)
        for field, value in changes.items():
            with self.subTest(field=field):
                current = record("b", "2026-10-06", 7, **{field: value})
                result = compare_analyses(previous, current)
                self.assertFalse(result["analysis_settings_match"])
                self.assertEqual(result["setting_differences"], [
                    {"field": field, "previous": previous[field], "current": value},
                ])
                self.assertEqual(result["count_delta"], 2)

    def test_comparison_does_not_modify_saved_records(self):
        previous = record("a", "2026-10-01", 5)
        current = record("b", "2026-10-06", 7)
        snapshots = previous.copy(), current.copy()
        compare_analyses(previous, current)
        self.assertEqual((previous, current), snapshots)
