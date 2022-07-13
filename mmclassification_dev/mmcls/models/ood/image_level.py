
from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter

from ..builder import OOD
from mmcls.models import build_classifier, build_ood_model

@OOD.register_module()
class MeanStdDetector(BaseModule):
    def __init__(self, ood_detector, crop_size, img_size, threshold,**kwargs):
        super(MeanStdDetector, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.ood_detector = build_ood_model(ood_detector)
        self.ood_detector.init_weights()
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
                crop = img[:, :, h:h + crop_size, w:w + crop_size]
                std, mean = torch.std_mean(crop, dim=(1, 2, 3))
                crops_mean.append(mean.unsqueeze(1))
                crops_std.append(std.unsqueeze(1))
            crops_mean = torch.cat(crops_mean, dim=1)
            crops_std = torch.cat(crops_std, dim=1)
            img_level_confs = torch.std(crops_mean, dim=1) + 3 * torch.std(crops_std, dim=1)
        ood_scores = self.ood_detector(**input)
        ood_scores[img_level_confs < self.threshold] -= 100
        return ood_scores

