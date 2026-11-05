import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from acne_care.dataset.dataset_v3 import build_dataset_v3


class DatasetV3Test(unittest.TestCase):
    def sample(self, root, split, name, seed, count=1):
        image_dir, label_dir = root / split / "images", root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        noise = random.Random(seed).randbytes(32 * 32 * 3)
        Image.frombytes("RGB", (32, 32), noise).save(image_dir / name)
        (label_dir / f"{Path(name).stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n" * count)

    def prepare(self, root):
        base, external = root / "base", root / "external"
        for index, split in enumerate(("train", "valid", "test")):
            self.sample(base, split, f"base-{index}.png", index)
            (external / split / "images").mkdir(parents=True)
            (external / split / "labels").mkdir()
        for index in range(20):
            self.sample(external, "train", f"external-{index}.png", index + 20, count=int(index % 4 != 0))
        return base, external

    def test_balances_external_only_and_preserves_original_splits_and_label_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, external = self.prepare(root)
            result = build_dataset_v3(base, external, root / "output")
            self.assertEqual(result["external_image_counts"], {"train": 14, "valid": 3, "test": 3})
            self.assertEqual(result["image_counts"], {"train": 15, "valid": 4, "test": 4})
            manifest = json.loads((root / "output/split-manifest.json").read_text())
            self.assertEqual(len(manifest["items"]), 23)
            for item in manifest["items"]:
                if item["origin"] == "baseline":
                    self.assertEqual(item["source_split"], item["split"])
                source = (base if item["origin"] == "baseline" else external) / item["source_label"]
                self.assertEqual(source.read_bytes(), (root / "output" / item["label_path"]).read_bytes())
            self.assertNotIn("path:", (root / "output/data.yaml").read_text())
            self.assertEqual(build_dataset_v3(base, external, root / "output"), result)

    def test_groups_reencoded_duplicates_and_same_source_derivatives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, external = self.prepare(root)
            self.sample(external, "test", "external-0.rf.11111111111111111111111111111111.png", 20, count=0)
            self.sample(external, "valid", "external-1.rf.11111111111111111111111111111111.png", 100, count=1)
            self.sample(external, "test", "external-1.rf.22222222222222222222222222222222.png", 101, count=1)
            build_dataset_v3(base, external, root / "output")
            rows = json.loads((root / "output/split-manifest.json").read_text())["items"]
            same_image = [r for r in rows if r["origin"] == "external" and r["image_sha256"] == hashlib.sha256((external / "train/images/external-0.png").read_bytes()).hexdigest()]
            self.assertEqual(len({r["split"] for r in same_image}), 1)
            same_source = [r for r in rows if "external-1.rf." in r["source_image"]]
            self.assertEqual(len({r["split"] for r in same_source}), 1)

    def test_external_match_is_anchored_to_existing_test_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, external = self.prepare(root)
            self.sample(external, "train", "matching-test.png", 2)
            result = build_dataset_v3(base, external, root / "output")
            self.assertEqual(result["external_reference_anchored_images"], 1)
            rows = json.loads((root / "output/split-manifest.json").read_text())["items"]
            matched = next(r for r in rows if r["source_image"].endswith("matching-test.png"))
            self.assertEqual(matched["split"], "test")

    def test_different_seed_does_not_overwrite_an_existing_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, external = self.prepare(root)
            build_dataset_v3(base, external, root / "output")
            with self.assertRaises(FileExistsError):
                build_dataset_v3(base, external, root / "output", seed=43)

    def test_rejects_source_overlap_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, external = self.prepare(root)
            with self.assertRaises(ValueError):
                build_dataset_v3(base, external, external / "output")

    def test_reencoded_image_is_grouped_by_perceptual_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, external = self.prepare(root)
            with Image.open(external / "train/images/external-0.png") as image:
                image.save(external / "test/images/reencoded.jpg", quality=95)
            (external / "test/labels/reencoded.txt").write_text("")
            build_dataset_v3(base, external, root / "output")
            rows = json.loads((root / "output/split-manifest.json").read_text())["items"]
            matched = [r for r in rows if r["source_image"] in ("train/images/external-0.png", "test/images/reencoded.jpg")]
            self.assertEqual(len({r["group_id"] for r in matched}), 1)

    def test_detects_modified_output_instead_of_overwriting_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, external = self.prepare(root)
            build_dataset_v3(base, external, root / "output")
            label = next((root / "output/train/labels").glob("*.txt"))
            label.write_text("manually modified")
            with self.assertRaisesRegex(ValueError, "differs from the source"):
                build_dataset_v3(base, external, root / "output")
            self.assertEqual(label.read_text(), "manually modified")

    def test_blocks_an_external_group_matching_conflicting_fixed_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, external = self.prepare(root)
            self.sample(external, "train", "base-0.png", 1)
            with self.assertRaisesRegex(ValueError, "multiple fixed baseline splits"):
                build_dataset_v3(base, external, root / "output")
            self.assertFalse((root / "output").exists())
