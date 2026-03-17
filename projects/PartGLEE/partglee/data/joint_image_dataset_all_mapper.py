# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging

import numpy as np
import torch
import re
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import pycocotools.mask as mask_utils
from fvcore.transforms.transform import HFlipTransform
from pycocotools import mask as coco_mask
from detectron2.structures import BitMasks, Instances, Boxes
import json
from torch.nn import functional as F
from .joint_image_dataset_LSJ_mapper import Joint_Image_LSJDatasetMapper
from .object_part_mapper import SemanticObjPartDatasetMapper
__all__ = ["Joint_Image_allDatasetMapper"]


class Joint_Image_allDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        
        self.inst_mapper = Joint_Image_LSJDatasetMapper(cfg, is_train)
        self.sem_seg_mapper = SemanticObjPartDatasetMapper(cfg, is_train)
    
    def __call__(self, dataset_dict):
        
        if 'is_sem_seg' in dataset_dict:
            if dataset_dict['is_sem_seg']:
                return self.sem_seg_mapper(dataset_dict)
            else:
                return self.inst_mapper(dataset_dict)
        else:
            return self.inst_mapper(dataset_dict)


    
       
