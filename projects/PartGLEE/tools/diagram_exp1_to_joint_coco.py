#!/usr/bin/env python3
"""Convert Experiment 1 diagram annotations to PartGLEE joint COCO format."""

import argparse
import json
import os
from pathlib import Path

from PIL import Image


DIAGRAM_OBJECT_CATEGORIES = [
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert diagram-labels-v3-standard.txt to bbox-only joint COCO JSON."
    )
    parser.add_argument("--input-json", required=True, help="Path to diagram-labels-v3-standard.txt")
    parser.add_argument("--output-dir", required=True, help="Directory for diagram_*_joint.json")
    parser.add_argument(
        "--old-prefix",
        default="",
        help="Optional prefix in JSON file_name values to replace for local image lookup.",
    )
    parser.add_argument(
        "--new-prefix",
        default="",
        help="Optional local prefix used with --old-prefix for image lookup.",
    )
    parser.add_argument(
        "--rewrite-output-paths",
        action="store_true",
        help="Write rewritten local image paths into output JSON instead of preserving input paths.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Keep entries with width/height from PIL unavailable. Intended only for debugging.",
    )
    return parser.parse_args()


def resolve_path(file_name, old_prefix, new_prefix):
    if old_prefix and new_prefix and file_name.startswith(old_prefix):
        return new_prefix + file_name[len(old_prefix):]
    return file_name


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def union_xyxy(boxes):
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def load_image_size(path):
    with Image.open(path) as image:
        return image.size


def build_categories(data):
    part_names = sorted(
        {
            part.strip()
            for item in data
            for part in (item.get("annotations") or {}).keys()
            if part and part.strip()
        }
    )
    categories = [
        {"id": idx, "name": name, "supercategory": "object"}
        for idx, name in enumerate(DIAGRAM_OBJECT_CATEGORIES)
    ]
    categories.extend(
        {
            "id": idx + len(DIAGRAM_OBJECT_CATEGORIES),
            "name": name,
            "supercategory": "part",
        }
        for idx, name in enumerate(part_names)
    )
    return categories, {name: idx + len(DIAGRAM_OBJECT_CATEGORIES) for idx, name in enumerate(part_names)}


def convert_split(data, categories, part_to_id, old_prefix, new_prefix, rewrite_output_paths, allow_missing, val_only):
    object_to_id = {name: idx for idx, name in enumerate(DIAGRAM_OBJECT_CATEGORIES)}
    images = []
    annotations = []
    ann_id = 1

    for item in data:
        image_id = int(item["image_id"])
        is_val = image_id % 10 >= 8
        if val_only and not is_val:
            continue

        file_name = item["file_name"]
        resolved_path = resolve_path(file_name, old_prefix, new_prefix)
        output_file_name = resolved_path if rewrite_output_paths else file_name
        if not os.path.exists(resolved_path):
            if not allow_missing:
                raise FileNotFoundError(f"Image not found for image_id={image_id}: {resolved_path}")
            width, height = 0, 0
        else:
            width, height = load_image_size(resolved_path)

        raw_annotations = item.get("annotations") or {}
        valid_parts = [
            (part.strip(), [float(v) for v in box])
            for part, box in raw_annotations.items()
            if part and part.strip() and isinstance(box, list) and len(box) == 4
        ]
        if not valid_parts:
            continue

        images.append(
            {
                "id": image_id,
                "file_name": output_file_name,
                "width": width,
                "height": height,
            }
        )

        super_category = item.get("super_category")
        if super_category not in object_to_id:
            raise ValueError(f"Unsupported super_category={super_category!r} for image_id={image_id}")

        object_box = union_xyxy([box for _, box in valid_parts])
        object_xywh = xyxy_to_xywh(object_box)
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": object_to_id[super_category],
                "bbox": object_xywh,
                "area": object_xywh[2] * object_xywh[3],
                "iscrowd": 0,
            }
        )
        ann_id += 1

        for part_name, box in valid_parts:
            part_xywh = xyxy_to_xywh(box)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": part_to_id[part_name],
                    "bbox": part_xywh,
                    "area": part_xywh[2] * part_xywh[3],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text())
    categories, part_to_id = build_categories(data)

    train = convert_split(
        data,
        categories,
        part_to_id,
        args.old_prefix,
        args.new_prefix,
        args.rewrite_output_paths,
        args.allow_missing_images,
        val_only=False,
    )
    val = convert_split(
        data,
        categories,
        part_to_id,
        args.old_prefix,
        args.new_prefix,
        args.rewrite_output_paths,
        args.allow_missing_images,
        val_only=True,
    )

    (output_dir / "diagram_train_joint.json").write_text(json.dumps(train))
    (output_dir / "diagram_val_joint.json").write_text(json.dumps(val))
    print(
        f"Wrote {len(train['images'])} train images, {len(val['images'])} val images, "
        f"{len(categories)} categories to {output_dir}"
    )


if __name__ == "__main__":
    main()
