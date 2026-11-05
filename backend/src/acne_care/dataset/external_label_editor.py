from __future__ import annotations

import csv
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image

from acne_care.dataset.external_candidate import IMAGE_SUFFIXES, SPLITS, audit_roboflow_csv_dataset
from acne_care.dataset.label_editor_dashboard import render_label_editor_html
from acne_care.dataset.label_review import build_label_review


def prepare_external_label_editor(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Prepare a separate, complete YOLO editing copy without approving any image."""
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("source and output directories must not overlap")
    report = audit_roboflow_csv_dataset(source_root)
    if not report.is_valid:
        raise ValueError("invalid candidate dataset: " + "; ".join(report.errors[:10]))
    if set(report.class_counts) - {"acne", "pimple", "spot"}:
        raise ValueError("unexpected source classes")
    images = sorted(path for split in SPLITS for path in (source_root / split).iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError("source dataset has no images")
    digest = sha256()
    for path in images + [source_root / split / "_annotations.csv" for split in SPLITS]:
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(sha256(path.read_bytes()).digest())
    signature = digest.hexdigest()
    if output_root.exists():
        manifest_path = output_root / "source-manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError("output already exists without a source manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_sha256") != signature:
            raise ValueError("source dataset changed; use a new output directory")
        document = json.loads((output_root / "audit.json").read_text())
        (output_root / "edit.html").write_text(render_label_editor_html(document, include_all=True), encoding="utf-8")
        return document

    rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for split in SPLITS:
        with (source_root / split / "_annotations.csv").open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                filename = row["filename"].strip()
                if Path(filename).name != filename or filename in {".", ".."}:
                    raise ValueError("source filenames must be plain basenames")
                rows_by_image[f"{split}/{filename}"].append(row)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        dataset = staging / "dataset"
        for split in SPLITS:
            (dataset / split / "images").mkdir(parents=True)
            (dataset / split / "labels").mkdir()
        dimensions = {}
        class_counts = {}
        for image in images:
            relative = image.relative_to(source_root)
            split = relative.parts[0]
            with Image.open(image) as opened:
                width, height = opened.size
                opened.verify()
            rows = rows_by_image[relative.as_posix()]
            lines = []
            for row in rows:
                if (int(row["width"]), int(row["height"])) != (width, height):
                    raise ValueError("CSV dimensions differ from the source image")
                if row["class"].strip() == "spot":
                    continue
                xmin, ymin, xmax, ymax = (int(row[field]) for field in ("xmin", "ymin", "xmax", "ymax"))
                lines.append(f"0 {(xmin+xmax)/2/width:.10f} {(ymin+ymax)/2/height:.10f} {(xmax-xmin)/width:.10f} {(ymax-ymin)/height:.10f}\n")
            shutil.copy2(image, dataset / split / "images" / image.name)
            (dataset / split / "labels" / f"{image.stem}.txt").write_text("".join(lines))
            key = f"{split}/images/{image.name}"
            dimensions[key] = (width, height)
            class_counts[key] = dict(Counter(row["class"].strip() for row in rows))
        document = build_label_review(dataset)
        document["dataset_root"] = str(output_root / "dataset")
        for item in document["items"]:
            key = item["image_path"]
            item["image_width"], item["image_height"] = dimensions[key]
            item["asset_path"] = f"dataset/{key}"
            item["source_class_counts"] = class_counts[key]
        if document["structural_errors"]:
            raise ValueError("converted dataset has structural label errors")
        (staging / "audit.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "source-manifest.json").write_text(json.dumps({
            "source_sha256": signature, "source_root": str(source_root),
            "class_mapping": {"acne": 0, "pimple": 0}, "excluded_classes": ["spot"],
            "training_approved": False,
        }, indent=2) + "\n")
        (staging / "edit.html").write_text(render_label_editor_html(document, include_all=True), encoding="utf-8")
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging)
        raise
    return document
