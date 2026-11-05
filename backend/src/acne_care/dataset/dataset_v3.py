from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from acne_care.dataset.audit import IMAGE_SUFFIXES, audit_yolo_dataset
from acne_care.dataset.external_candidate import _perceptual_fingerprint, _roboflow_source_name


SPLITS = ("train", "valid", "test")
RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}


def _targets(size: int) -> dict[str, int]:
    result = {split: int(size * RATIOS[split]) for split in SPLITS}
    remaining = size - sum(result.values())
    order = sorted(SPLITS, key=lambda split: (-(size * RATIOS[split] - result[split]), SPLITS.index(split)))
    for split in order[:remaining]:
        result[split] += 1
    return result


def _inventory(root: Path, origin: str) -> list[dict[str, Any]]:
    audit = audit_yolo_dataset(root)
    # Old external partitions may contain duplicates: regroup those before writing.
    errors = [error for error in audit.errors if not (origin == "external" and error.startswith("exact duplicate crosses splits:"))]
    if errors:
        raise ValueError(f"invalid {origin} dataset: " + "; ".join(errors[:5]))
    if set(audit.class_ids) - {0}:
        raise ValueError("only the single Acne class is supported")
    review_path = root / "label-edits-source.json"
    reviews = json.loads(review_path.read_text()).get("edits", {}) if review_path.is_file() else {}
    records = []
    for split in SPLITS:
        for image in sorted((root / split / "images").iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = root / split / "labels" / f"{image.stem}.txt"
            relative = image.relative_to(root).as_posix()
            annotations = len([line for line in label.read_text().splitlines() if line.strip()])
            reviewed = bool(reviews.get(relative, {}).get("completed")) if origin == "external" and review_path.is_file() else None
            records.append({
                "origin": origin, "source_split": split, "source_image": relative,
                "source_label": label.relative_to(root).as_posix(),
                "image_sha256": sha256(image.read_bytes()).hexdigest(),
                "label_sha256": sha256(label.read_bytes()).hexdigest(),
                "source_key": _roboflow_source_name(image),
                "perceptual_hashes": _perceptual_fingerprint(image),
                "annotations": annotations, "human_reviewed": reviewed,
            })
    return records


def _groups(records: list[dict[str, Any]], distance: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    counts = Counter()
    for right, b in enumerate(records):
        if b["origin"] != "external":
            continue
        for left in range(right):
            a = records[left]
            if a["image_sha256"] == b["image_sha256"]:
                reason = "exact_image_match"
            elif a["source_key"] == b["source_key"]:
                reason = "same_source_name"
            elif all((x ^ y).bit_count() <= distance for x, y in zip(a["perceptual_hashes"], b["perceptual_hashes"])):
                reason = "perceptual_match"
            else:
                continue
            parent[find(right)] = find(left)
            counts[reason] += 1
    components = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    groups = []
    for members in components.values():
        external = [i for i in members if records[i]["origin"] == "external"]
        if not external:
            continue
        anchors = {records[i]["source_split"] for i in members if records[i]["origin"] == "baseline"}
        if len(anchors) > 1:
            raise ValueError("external duplicate group matches multiple fixed baseline splits")
        group_id = sha256(json.dumps(sorted(f"{records[i]['origin']}:{records[i]['source_image']}" for i in members)).encode()).hexdigest()[:16]
        strata = Counter(_stratum(records[i]) for i in external)
        groups.append({"id": group_id, "members": external, "anchor": next(iter(anchors), None), "strata": strata})
    return groups, dict(counts)


def _stratum(record: dict[str, Any]) -> str:
    count = record["annotations"]
    density = "empty" if count == 0 else "1-3" if count < 4 else "4-9" if count < 10 else "10+"
    return f"reviewed={record['human_reviewed']};boxes={density}"


def _assign(groups: list[dict[str, Any]], target: dict[str, int], seed: int) -> dict[int, tuple[str, str]]:
    assignments = {}
    used = Counter()
    strata = Counter()
    for group in groups:
        strata.update(group["strata"])
    stratum_targets = {key: _targets(size) for key, size in strata.items()}
    stratum_used = {key: Counter() for key in strata}
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda group: (group["anchor"] is None, -len(group["members"])))
    for group in ordered:
        size = len(group["members"])
        if group["anchor"]:
            split = group["anchor"]
        else:
            candidates = [split for split in SPLITS if used[split] + size <= target[split]] or list(SPLITS)

            def score(split):
                improvement = sum(
                    count * (stratum_targets[key][split] - stratum_used[key][split]) / max(stratum_targets[key][split], 1)
                    for key, count in group["strata"].items()
                ) / size
                return improvement, (target[split] - used[split]) / max(target[split], 1), -SPLITS.index(split)

            split = max(candidates, key=score)
        used[split] += size
        for key, count in group["strata"].items():
            stratum_used[key][split] += count
        for index in group["members"]:
            assignments[index] = split, group["id"]
    return assignments


def _verify(root: Path, items: list[dict[str, Any]]) -> None:
    group_splits = defaultdict(set)
    for item in items:
        for kind in ("image", "label"):
            path = root / item[f"{kind}_path"]
            if sha256(path.read_bytes()).hexdigest() != item[f"{kind}_sha256"]:
                raise ValueError(f"copied {kind} differs from the source")
        if item["origin"] == "baseline" and item["split"] != item["source_split"]:
            raise ValueError("baseline split changed")
        if item["origin"] == "external":
            group_splits[item["group_id"]].add(item["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("an external group crosses splits")
    report = audit_yolo_dataset(root)
    if report.errors:
        raise ValueError("invalid output dataset: " + "; ".join(report.errors[:5]))


def build_dataset_v3(
    baseline_root: Path, external_root: Path, output_root: Path, *, seed: int = 42, max_hash_distance: int = 6,
) -> dict[str, Any]:
    baseline_root, external_root, output_root = baseline_root.resolve(), external_root.resolve(), output_root.resolve()
    roots = (baseline_root, external_root, output_root)
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError("dataset directories must not overlap")
    if not 0 <= max_hash_distance <= 64:
        raise ValueError("hash distance must be between 0 and 64")
    records = _inventory(baseline_root, "baseline") + _inventory(external_root, "external")
    external_size = sum(record["origin"] == "external" for record in records)
    if external_size < 10:
        raise ValueError("at least ten external images are required for three splits")
    signature = sha256(json.dumps({"records": records, "seed": seed, "distance": max_hash_distance, "ratios": RATIOS}, sort_keys=True).encode()).hexdigest()
    if output_root.exists():
        path = output_root / "split-manifest.json"
        if not path.is_file():
            raise FileExistsError("output exists without a split manifest")
        manifest = json.loads(path.read_text())
        if manifest.get("input_signature") != signature:
            raise FileExistsError("inputs changed; use a new dataset version directory")
        _verify(output_root, manifest["items"])
        return json.loads((output_root / "build-summary.json").read_text())

    groups, matches = _groups(records, max_hash_distance)
    targets = _targets(external_size)
    assignments = _assign(groups, targets, seed)
    if set(assignments) != {i for i, record in enumerate(records) if record["origin"] == "external"}:
        raise ValueError("not all external images were assigned")
    items = []
    for index, record in enumerate(records):
        split, group_id = assignments[index] if record["origin"] == "external" else (record["source_split"], None)
        filename = f"{record['origin']}__{record['source_split']}__{Path(record['source_image']).name}"
        items.append({**record, "split": split, "group_id": group_id,
                      "image_path": f"{split}/images/{filename}",
                      "label_path": f"{split}/labels/{Path(filename).stem}.txt"})
    image_counts = {split: sum(item["split"] == split for item in items) for split in SPLITS}
    external_counts = {split: sum(item["split"] == split and item["origin"] == "external" for item in items) for split in SPLITS}
    summary = {
        "schema_version": 1, "seed": seed, "external_target_ratios": RATIOS,
        "external_target_counts": targets, "external_image_counts": external_counts, "image_counts": image_counts,
        "baseline_image_counts": {split: image_counts[split] - external_counts[split] for split in SPLITS},
        "annotation_counts": {split: sum(item["annotations"] for item in items if item["split"] == split) for split in SPLITS},
        "external_annotation_counts": {split: sum(item["annotations"] for item in items if item["split"] == split and item["origin"] == "external") for split in SPLITS},
        "external_human_reviewed_counts": {split: sum(item["human_reviewed"] is True for item in items if item["split"] == split and item["origin"] == "external") for split in SPLITS},
        "external_empty_label_counts": {split: sum(item["annotations"] == 0 for item in items if item["split"] == split and item["origin"] == "external") for split in SPLITS},
        "external_groups": len(groups), "external_multi_image_groups": sum(len(group["members"]) > 1 for group in groups),
        "external_reference_anchored_images": sum(len(group["members"]) for group in groups if group["anchor"]),
        "match_pairs": matches, "max_hash_distance": max_hash_distance,
        "external_group_cross_split_violations": 0,
        "person_disjoint_verified": False, "all_external_labels_human_reviewed": False,
        "notes": ["Source names and image hashes cannot guarantee separation of every person across splits.",
                  "Unreviewed labels, including empty labels, are retained as explicitly requested.",
                  "Evaluate the control and new models on identical original, external and combined subsets."],
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        for split in SPLITS:
            (staging / split / "images").mkdir(parents=True)
            (staging / split / "labels").mkdir()
        for item in items:
            source = baseline_root if item["origin"] == "baseline" else external_root
            shutil.copy2(source / item["source_image"], staging / item["image_path"])
            shutil.copy2(source / item["source_label"], staging / item["label_path"])
        (staging / "data.yaml").write_text('train: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: "Acne"\n')
        (staging / "split-manifest.json").write_text(json.dumps({"schema_version": 1, "input_signature": signature, "items": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "build-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _verify(staging, items)
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging)
        raise
    return summary
