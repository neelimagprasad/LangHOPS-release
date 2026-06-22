import json
import os
from collections import defaultdict

from detectron2.data import DatasetCatalog, MetadataCatalog


PARTIMAGENET_SUPER_CATEGORIES = [
    "Quadruped",
    "Biped",
    "Fish",
    "Bird",
    "Snake",
    "Reptile",
    "Car",
    "Bicycle",
    "Boat",
    "Aeroplane",
    "Bottle",
]


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def _load_canonical_parts(labels_file):
    dataset_images = _load_json(labels_file, [])
    parts = set()
    part_to_super = defaultdict(set)

    for img_data in dataset_images:
        super_category = img_data.get("super_category")
        if super_category not in PARTIMAGENET_SUPER_CATEGORIES:
            continue
        for part_name in (img_data.get("annotations") or {}).keys():
            if part_name and part_name.strip():
                cleaned = part_name.strip()
                parts.add(cleaned)
                part_to_super[cleaned].add(super_category)

    return sorted(parts), part_to_super


def _get_part_label(record):
    for key in ("label", "part_label", "part_name"):
        value = record.get(key)
        if value and value.strip():
            return value.strip()
    return None


def _load_part_classification_dicts(json_file, labels_file, split):
    if not os.path.exists(json_file):
        raise FileNotFoundError(
            f"Missing Exp2 part classification JSON: {json_file}. "
            "Copy diagram_parts_v3_exp2.json into DETECTRON2_DATASETS/diagram/."
        )
    if not os.path.exists(labels_file):
        raise FileNotFoundError(
            f"Missing Exp1 canonical labels file: {labels_file}. "
            "Copy diagram-labels-v3-standard.txt into DETECTRON2_DATASETS/diagram/."
        )

    canonical_parts, part_to_super = _load_canonical_parts(labels_file)
    data = _load_json(json_file, [])
    if not canonical_parts:
        canonical_parts = sorted({_get_part_label(item) for item in data if _get_part_label(item)})
    part_to_id = {part: idx for idx, part in enumerate(canonical_parts)}

    records = []
    missing_files = 0
    skipped_labels = 0

    for item in data:
        image_id = item.get("image_id")
        if image_id is None:
            continue
        is_train = image_id % 10 < 8
        if split == "train" and not is_train:
            continue
        if split == "val" and is_train:
            continue

        file_name = item.get("file_name")
        part_label = _get_part_label(item)
        if not file_name or not part_label:
            continue
        if part_label not in part_to_id:
            skipped_labels += 1
            continue
        if not os.path.exists(file_name):
            missing_files += 1
            if missing_files <= 5:
                print(f"Warning: part classification image not found: {file_name}")
            continue

        super_category = item.get("super_category")
        if super_category not in PARTIMAGENET_SUPER_CATEGORIES:
            candidates = part_to_super.get(part_label, set())
            super_category = sorted(candidates)[0] if candidates else "unknown"

        super_category_id = (
            PARTIMAGENET_SUPER_CATEGORIES.index(super_category)
            if super_category in PARTIMAGENET_SUPER_CATEGORIES
            else None
        )
        records.append(
            {
                "file_name": file_name,
                "image_id": image_id,
                "height": 256,
                "width": 256,
                "task": "part_classification",
                "dataset_name": "part_classification",
                "part_label": part_label,
                "part_id": part_to_id[part_label],
                "super_category": super_category,
                "super_category_id": super_category_id,
            }
        )

    print(
        f"Loaded {len(records)} records for part_classification_{split} "
        f"(missing_files={missing_files}, skipped_labels={skipped_labels})"
    )
    return records


def _get_part_classification_metadata(labels_file, json_file):
    canonical_parts, _ = _load_canonical_parts(labels_file)
    if not canonical_parts:
        data = _load_json(json_file, [])
        canonical_parts = sorted({_get_part_label(item) for item in data if _get_part_label(item)})
    return {
        "thing_classes": canonical_parts,
        "super_categories": PARTIMAGENET_SUPER_CATEGORIES,
        "evaluator_type": "coco",
    }


def register_all_diagram_part_classification(root):
    json_file = os.path.join(root, "diagram/diagram_parts_v3_exp2.json")
    labels_file = os.path.join(root, "diagram/diagram-labels-v3-standard.txt")
    metadata = _get_part_classification_metadata(labels_file, json_file)

    for split in ("train", "val"):
        DatasetCatalog.register(
            f"part_classification_{split}",
            lambda split=split: _load_part_classification_dicts(json_file, labels_file, split),
        )
        MetadataCatalog.get(f"part_classification_{split}").set(**metadata)
