# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data.datasets.register_coco import register_coco_instances


DIAGRAM_OBJECT_CATEGORIES = [
    {"id": 0, "name": "Quadruped"},
    {"id": 1, "name": "Biped"},
    {"id": 2, "name": "Fish"},
    {"id": 3, "name": "Bird"},
    {"id": 4, "name": "Snake"},
    {"id": 5, "name": "Reptile"},
    {"id": 6, "name": "Car"},
    {"id": 7, "name": "Bicycle"},
    {"id": 8, "name": "Boat"},
    {"id": 9, "name": "Aeroplane"},
    {"id": 10, "name": "Bottle"},
]

_DIAGRAM_JOINT = {
    "diagram_joint_train": ("", "diagram/diagram_train_joint.json"),
    "diagram_joint_val": ("", "diagram/diagram_val_joint.json"),
}


def _load_categories(json_file):
    if not os.path.exists(json_file):
        # The converter writes categories into the JSON. This fallback keeps
        # registration importable before dataset preparation has run.
        return DIAGRAM_OBJECT_CATEGORIES + [
            {"id": idx + len(DIAGRAM_OBJECT_CATEGORIES), "name": f"diagram_part_{idx}"}
            for idx in range(560)
        ]
    with open(json_file, "r") as f:
        return json.load(f)["categories"]


def _get_diagram_metadata(json_file):
    categories = _load_categories(json_file)
    id_to_name = {x["id"]: x["name"] for x in categories}
    thing_dataset_id_to_contiguous_id = {x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    obj_ids = [
        thing_dataset_id_to_contiguous_id[item["id"]]
        for item in DIAGRAM_OBJECT_CATEGORIES
    ]

    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "obj_ids": obj_ids,
        "part_ids": [
            thing_dataset_id_to_contiguous_id[item["id"]]
            for item in categories
            if thing_dataset_id_to_contiguous_id[item["id"]] not in obj_ids
        ],
    }


def register_all_diagram_joint(root):
    for key, (image_root, json_file) in _DIAGRAM_JOINT.items():
        full_json_file = os.path.join(root, json_file) if "://" not in json_file else json_file
        register_coco_instances(
            key,
            _get_diagram_metadata(full_json_file),
            full_json_file,
            os.path.join(root, image_root),
            dataset_name_in_dict="diagram_joint",
            evaluator_type="coco",
        )
