
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
class MeanStdDetector(BaseModule):
    def __init__(self, crop_size, img_size, threshold, ood_detector=None, **kwargs):
        super(MeanStdDetector, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.has_ood_detector = True if ood_detector else False
        if self.has_ood_detector:
            self.ood_detector = build_ood_model(ood_detector)
        else:
            self.ood_detector = no_ood_detector
        self.crop_size = crop_size
        self.img_size = img_size
        self.threshold = threshold

    def forward(self, **input):
        with torch.no_grad():
            img = input['img']
            crop_size = self.crop_size
            img_size = self.img_size
            crops_mean = []
            crops_std = []
            corner_list = []
            for h in range(img_size // crop_size):
                for w in range(img_size // crop_size):
                    corner_list.append([h * crop_size, w * crop_size])
            for h, w in corner_list:
                crop = img[:, :, h: h + crop_size, w: w + crop_size]
                std, mean = torch.std_mean(crop, dim=(1, 2, 3))
                crops_mean.append(mean.unsqueeze(1))
                crops_std.append(std.unsqueeze(1))
            crops_mean = torch.cat(crops_mean, dim=1)
            crops_std = torch.cat(crops_std, dim=1)
            img_level_confs = torch.std(crops_mean, dim=1) + 3 * torch.std(crops_std, dim=1)
        if self.has_ood_detector:
            ood_scores = self.ood_detector(**input)
            # ood_scores = ood_scores - ood_scores.min()
            # ood_scores[img_level_confs < self.threshold] *= 0.5
            img_level_confs = ((1/self.threshold)**0.5) * torch.pow(img_level_confs, 0.5)
            img_level_confs[img_level_confs > 1] = 1
            ood_scores *= img_level_confs
        else:
            ood_scores = img_level_confs
        return ood_scores

