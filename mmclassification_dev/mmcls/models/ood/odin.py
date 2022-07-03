from mmcv.runner import BaseModule
import torch
import os
import numpy as np
from collections import Counter

from ..builder import OOD
from mmcls.models import build_classifier


@OOD.register_module()
class ODIN(BaseModule):
    def __init__(self, classifier, num_classes, temperature, epsilon, **kwargs):
        super(ODIN, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
        self.epsilon = epsilon
        self.criterion = torch.nn.CrossEntropyLoss().to("cuda:{}".format(self.local_rank))


    def forward(self, **input):
        x = input['img']
        # self.classifier.zero_grad()
        outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / self.temperature
        labels = torch.LongTensor(maxIndexTemp).cuda()
        loss = self.criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -self.epsilon, gradient)
        outputs = self.classifier(tempInputs)
        outputs = outputs / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs = np.max(nnOutputs, axis=1)
        confs = torch.tensor(confs)
        return confs

@OOD.register_module()
class ODINCustom(BaseModule):
    def __init__(self, classifier, num_classes, temperature, target_file=None,**kwargs):
        super(ODINCustom, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        classifier['head']['require_features'] = True
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
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
            U = torch.norm(features, p=1, dim=1)
            out_softmax = torch.nn.functional.softmax(outputs, dim=1)
            targets = self.target
            V = torch.norm((targets - out_softmax), p=1, dim=1)
            S = U * V / 2048
        return S




