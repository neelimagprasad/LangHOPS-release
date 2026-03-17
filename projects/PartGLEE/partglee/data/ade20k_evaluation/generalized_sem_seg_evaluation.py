# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
from torch.nn import functional as F
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.evaluation import SemSegEvaluator
# from utils.visualizer import ColorMode, Visualizer

# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import SemSegEvaluator
from .custom_visualizer import CustomVisualizer

class GeneralizedSemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
        visualize=False,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )
        self.meta = meta
        
        self.visualize = visualize
        self.vis_path = os.path.join(output_dir, "visualization")
        if self.visualize and not os.path.isdir(self.vis_path):
            os.makedirs(self.vis_path)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            obj_mask = input["instances"].gt_masks[0]
            # print("output['sem_seg'].shape: ", output['sem_seg'].shape)
            
            output = self.post_process_func(
                output["sem_seg"], image=np.array(Image.open(input["file_name"]))
            )
            # print("output.shape: ", output.shape)
            output = output.argmax(dim=0).to(self._cpu_device)
            # print("output.shape: ", output.shape)
            obj_mask = F.interpolate(obj_mask.float().unsqueeze(0).unsqueeze(0), size=output.shape[-2:], mode='nearest').squeeze()
            output[obj_mask==0.0] = self.meta.ignore_label
            
            pred = np.array(output, dtype=np.int64)
            gt_classes = input["obj_part_instances"].gt_classes
            gt_masks = input["obj_part_instances"].gt_masks
            if len(gt_masks) == 0:
                gt = np.zeros_like(pred) + self._ignore_label
            else:
                gt = np.zeros_like(gt_masks[0], dtype=np.float64) + self._ignore_label
                for i in range(len(gt_classes)):
                    gt[gt_masks[i] == True] = gt_classes[i]
                eval_image_size = pred.shape[-2:]
                gt = F.interpolate(torch.tensor(gt).unsqueeze(0).unsqueeze(0), size=eval_image_size, mode='nearest').squeeze()
                gt = gt.int().numpy()
            # print("gt_classes: ", gt_classes)
            # print("len(gt_masks): ", len(gt_masks))
            # print("self._conf_matrix: ", self._conf_matrix.shape)
            # print("np.unique(pred): ", np.unique(pred))
            # print("np.unique(gt): ", np.unique(gt))
            # print("pred.shape: ", pred.shape)
            # print("gt.shape: ", gt.shape)
            # print("self._ignore_label: ", self._ignore_label)
            # print("self._num_classes: ", self._num_classes)
            pred[pred == self._ignore_label] = self._num_classes
            gt[gt == self._ignore_label] = self._num_classes
            # print("np.unique((self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1)): ", np.unique((self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1)))
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
            
            if self.visualize:
                ext = os.path.splitext(input["file_name"])[1]
                input_img_tensor = F.interpolate(input["image"].unsqueeze(0), size=eval_image_size, mode='bilinear').squeeze()
                input_img_npy = input_img_tensor.permute(1, 2, 0).int().numpy()

                visualizer_pred = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                visualizer_gt = CustomVisualizer(input_img_npy, self.meta, instance_mode=ColorMode.SEGMENTATION)
                
                vis_pred = visualizer_pred.draw_sem_seg(pred)
                vis_pred.save(os.path.join(self.vis_path, os.path.basename(input["file_name"])))

                vis_gt = visualizer_gt.draw_sem_seg(gt)
                vis_gt.save(os.path.join(self.vis_path, os.path.basename(input["file_name"]).replace(ext, "_gt.jpg")))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float64)
        iou = np.full(self._num_classes, np.nan, dtype=np.float64)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        # res["mACC"] = 100 * macc
        # res["pACC"] = 100 * pacc
        # for i, name in enumerate(self._class_names):
        #     res["ACC-{}".format(name)] = 100 * acc[i]
        if self._evaluation_set is not None:
            for set_name, set_inds in self._evaluation_set.items():
                iou_list = []
                set_inds = np.array(set_inds, np.int64)
                mask = np.zeros((len(iou),)).astype(np.bool_)
                mask[set_inds] = 1
                miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
                pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
                res["mIoU-{}".format(set_name)] = 100 * miou
                # res["pAcc-{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
                miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
                pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
                res["mIoU-un{}".format(set_name)] = 100 * miou
                # res["pAcc-un{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
        res['h-IoU'] = 2 * (res['mIoU-base'] * res['mIoU-unbase']) / (res['mIoU-base'] + res['mIoU-unbase'])
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
