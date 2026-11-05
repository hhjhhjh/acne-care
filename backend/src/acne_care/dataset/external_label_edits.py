from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from acne_care.dataset.label_edits import _format_annotation, build_dataset_v2
from acne_care.dataset.label_review_dashboard import review_dataset_fingerprint


def _box_key(annotation: dict[str, Any]) -> tuple:
    return (annotation["class_id"], *(round(float(annotation[key]), 8)
            for key in ("x_center", "y_center", "width", "height")))


def apply_completed_external_edits(editor_root: Path, edits_path: Path, output_root: Path) -> dict[str, Any]:
    """Apply only explicitly completed edits; preserve every other source label byte-for-byte."""
    editor_root, output_root = editor_root.resolve(), output_root.resolve()
    if editor_root == output_root or editor_root in output_root.parents or output_root in editor_root.parents:
        raise ValueError("editor and output directories must not overlap")
    audit = json.loads((editor_root / "audit.json").read_text())
    source = (editor_root / "dataset").resolve()
    if Path(audit["dataset_root"]).resolve() != source:
        raise ValueError("audit dataset root differs from editor dataset")
    raw_bytes = edits_path.read_bytes()
    export = json.loads(raw_bytes)
    if export.get("schema_version") != 1 or export.get("review_splits") != ["train", "valid", "test"]:
        raise ValueError("expected an all-image external label editor export")
    fingerprint = review_dataset_fingerprint(audit, splits=("train", "valid", "test"))
    if export.get("dataset_fingerprint") != fingerprint:
        raise ValueError("label edits belong to a different dataset version")
    edits = export.get("edits")
    if not isinstance(edits, dict):
        raise ValueError("edits must be an object")
    by_path = {item["image_path"]: item for item in audit["items"]}
    if set(edits) - set(by_path):
        raise ValueError("unknown edited image")
    for path, edit in edits.items():
        if not isinstance(edit, dict) or type(edit.get("completed")) is not bool or type(edit.get("excluded")) is not bool:
            raise ValueError("completed and excluded must be boolean values")
    completed = {path: edit for path, edit in edits.items() if edit["completed"]}
    if not completed:
        raise ValueError("no completed images to apply")
    for edit in completed.values():
        if not isinstance(edit.get("annotations"), list):
            raise ValueError("annotations must be a list")
        for annotation in edit["annotations"]:
            if annotation.get("class_id") != 0:
                raise ValueError("only class 0 is supported")
            _format_annotation(annotation)

    digest = sha256()
    for path, item in sorted(by_path.items()):
        relative = Path(path)
        if len(relative.parts) != 3 or relative.parts[0] not in {"train", "valid", "test"} or relative.parts[1] != "images":
            raise ValueError("unexpected audit image path")
        for file in (source / path, source / item["label_path"]):
            if source not in file.resolve().parents:
                raise ValueError("audit path escapes source dataset")
            digest.update(file.relative_to(source).as_posix().encode())
            digest.update(sha256(file.read_bytes()).digest())
    source_hash = digest.hexdigest()
    edits_hash = sha256(raw_bytes).hexdigest()
    if output_root.exists():
        summary_path = output_root / "partial-review-summary.json"
        if not summary_path.is_file():
            raise FileExistsError("output already exists; choose a new output directory")
        summary = json.loads(summary_path.read_text())
        if summary.get("edits_sha256") != edits_hash or summary.get("source_dataset_sha256") != source_hash:
            raise FileExistsError("output belongs to different inputs; choose a new output directory")
        return summary

    added = removed = changed = 0
    for path, edit in completed.items():
        before = Counter(_box_key(box) for box in by_path[path]["annotations"])
        after = Counter(_box_key(box) for box in edit["annotations"]) if not edit["excluded"] else Counter()
        added += sum((after - before).values())
        removed += sum((before - after).values())
        changed += int(before != after and not edit["excluded"])
    completed_by_split = Counter(by_path[path]["split"] for path in completed)
    result = build_dataset_v2(source, output_root, audit, {**export, "edits": completed})
    # Verify the preservation contract after building the independent copy.
    for path, item in by_path.items():
        if path in completed:
            continue
        label = item["label_path"]
        if (source / label).read_bytes() != (output_root / label).read_bytes():
            raise ValueError("an unreviewed label changed during the build")
    summary = {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "edits_sha256": edits_hash,
        "source_dataset_sha256": source_hash,
        "completed_images": len(completed),
        "completed_images_by_split": {split: completed_by_split[split] for split in ("train", "valid", "test")},
        "unreviewed_images_preserved": len(by_path) - len(completed),
        "changed_label_images": changed,
        "reviewed_without_label_changes": len(completed) - changed - result.excluded_images,
        "added_boxes": added,
        "removed_boxes": removed,
        "excluded_images": result.excluded_images,
        "image_counts": result.image_counts,
        "annotation_counts": result.annotation_counts,
        "all_images_human_reviewed": False,
        "unreviewed_policy": "preserve original converted labels; do not mark as reviewed",
        "class_mapping": {"acne": 0, "pimple": 0},
        "excluded_source_classes": ["spot"],
    }
    (output_root / "label-edits-source.json").write_bytes(raw_bytes)
    (output_root / "partial-review-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
