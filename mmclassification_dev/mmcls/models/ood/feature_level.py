from mmcv.runner import BaseModule    # noqa
import torch
import os
import numpy as np  # noqa
from collections import Counter  # noqa

from ..builder import OOD
from mmcls.models import build_classifier, build_ood_model    # noqa


def no_ood_detector(**kwargs):
    raise RuntimeError("No Feature-level OOD Detector Configured!")

@OOD.register_module()
class PatchSim(BaseModule):
    def __init__(self, num_crop, img_size, threshold, order=1, ood_detector=None, **kwargs):
        super(PatchSim, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.has_ood_detector = True if ood_detector else False
        if self.has_ood_detector:
            self.ood_detector = build_ood_model(ood_detector)
        else:
            self.ood_detector = no_ood_detector
        self.num_crop = num_crop
        self.img_size = img_size
        self.threshold = threshold
        self.order = order


    def forward(self, **input):
        if "type" in input:
            type = input['type']
            del input['type']
        with torch.no_grad():
            img = input['img']
            img_size = self.img_size
            crop_size = int(img_size / self.num_crop)
            corner_list = []
            crops = []
            for h in range(self.num_crop):
                for w in range(self.num_crop):
                    corner_list.append([h * crop_size, w * crop_size])
            for h, w in corner_list:
                crop = img[:, :, h: h + crop_size, w: w + crop_size]
                input['img'] = crop
                _, crop_feature = self.ood_detector.classifier(return_loss=False, softmax=False,
                                                               post_process=False, require_features=True, **input)
                crops.append(crop_feature)
            input['img'] = img
            patch_sim = 0
            for i in range(len(crops)-1):
                for j in range(i+1, len(crops)):
                    patch_sim += - crops[i] * crops[j] / (torch.norm(crops[i], dim=1) * torch.norm(crops[j], dim=1))
        return patch_sim, type

