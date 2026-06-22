"""
Dataset mapper for Experiment 2 part classification.

Each image contains a single cropped part. We represent it as one instance with
a full-image bounding box and the part class as `gt_classes`.
"""

import copy
import logging
import os

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, Instances


logger = logging.getLogger(__name__)


class PartClassificationDatasetMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT
        self.debug_samples_remaining = int(os.getenv("LANGHOPS_DEBUG_DATASET_SAMPLES", "0"))

        if is_train:
            aug_list = []
            if cfg.INPUT.MIN_SIZE_TRAIN:
                aug_list.append(
                    T.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TRAIN,
                        cfg.INPUT.MAX_SIZE_TRAIN,
                        sample_style="choice",
                    )
                )
            if cfg.INPUT.RANDOM_FLIP != "none":
                aug_list.append(T.RandomFlip(horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal"))
            self.augmentations = T.AugmentationList(aug_list)
        else:
            self.augmentations = T.AugmentationList(
                [
                    T.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TEST,
                        cfg.INPUT.MAX_SIZE_TEST,
                        sample_style="choice",
                    )
                ]
            )

        logger.info("PartClassificationDatasetMapper initialized (is_train=%s)", is_train)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # Exp2 registration uses placeholder dimensions for speed; cropped part
        # images have varied real sizes, so update the record from the image.
        dataset_dict["height"], dataset_dict["width"] = image.shape[:2]

        aug_input = T.AugInput(image)
        self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        part_id = dataset_dict.get("part_id")
        if part_id is None:
            raise ValueError(f"Missing part_id for {dataset_dict.get('file_name')}")

        height, width = image_shape
        instances = Instances(image_shape)
        instances.gt_boxes = Boxes(torch.tensor([[0.0, 0.0, float(width), float(height)]], dtype=torch.float32))
        instances.gt_classes = torch.tensor([part_id], dtype=torch.int64)

        dataset_dict["task"] = "part_classification"
        dataset_dict["dataset_name"] = "part_classification"
        dataset_dict["instances"] = instances

        if self.debug_samples_remaining > 0:
            print(
                "[PartClassificationMapperDebug] "
                f"file_name={dataset_dict.get('file_name')} "
                f"image_id={dataset_dict.get('image_id')} "
                f"part_label={dataset_dict.get('part_label')} "
                f"part_id={part_id} "
                f"image_shape={image_shape}",
                flush=True,
            )
            self.debug_samples_remaining -= 1

        return dataset_dict
