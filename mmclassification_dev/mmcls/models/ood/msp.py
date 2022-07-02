from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter

from ..builder import OOD
from mmcls.models import build_classifier


@OOD.register_module()
class MSP(BaseModule):
    def __init__(self, classifier, **kwargs):
        super(MSP, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()

    def forward(self, **input):
        with torch.no_grad():
            outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            confs, _ = torch.max(out_softmax)
        return confs

@OOD.register_module()
class MSPCustom(BaseModule):
    def __init__(self, classifier, num_classes, target_file=None,**kwargs):
        super(MSPCustom, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        classifier['head']['require_features'] = True
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        if target_file is not None:
            cls_idx = []
            with open(target_file, 'r') as f:
                for line in f.readlines():
                    segs = line.strip().split(' ')
                    cls_idx.append(int(segs[-1]))
            cls_idx = np.array(cls_idx, dtype='int')
            label_stat = Counter(cls_idx)
            cls_num = [-1 for _ in range(num_classes)]
            for i in range(num_classes):
                cat_num = int(label_stat[i])
                cls_num[i] = cat_num
            target = cls_num / np.sum(cls_num)
            self.target = torch.tensor(target).to("cuda:{}".format(self.local_rank)).unsqueeze(0)
        else:
            self.target = torch.ones((1, self.num_classes)).to("cuda:{}".format(self.local_rank)) / self.num_classes

    def forward(self, **input):
        with torch.no_grad():
            outputs, features = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            targets = self.target
            confs = out_softmax - targets
            confs, _ = torch.max(confs)
        return confs




