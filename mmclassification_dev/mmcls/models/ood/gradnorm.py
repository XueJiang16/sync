from mmcv.runner import BaseModule
import torch
import os

from ..builder import OOD
from mmcls.models import build_classifier


@OOD.register_module()
class GradNorm(BaseModule):
    def __init__(self, classifier, num_classes, temperature, **kwargs):
        super(GradNorm, self).__init__()
        self.local_rank = os.environ['LOCAL_RANK']
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1).to("cuda:{}".format(self.local_rank))

    def forward(self, **input):
        self.classifier.eval()
        self.classifier.zero_grad()
        img = input['img']
        assert img.shape[0] == 1, "GradNorm backward implementation only supports batch = 1."
        outputs = self.classifier(return_loss=False, softmax=False, post_process=False, **input)
        # print("Self rank: {}, output device = {}".format(self.local_rank, outputs.device))
        # assert False
        # outputs, _ = self.classifier.simple_test(softmax=False, **input)
        targets = torch.ones((img.shape[0], self.num_classes)).to("cuda:{}".format(self.local_rank))
        outputs = outputs / self.temperature
        print(outputs)
        assert False
        loss = torch.sum(torch.mean(-targets * self.logsoftmax(outputs), dim=-1))

        loss.backward()
        layer_grad = self.classifier.head.fc.weight.grad.data
        layer_grad_norm = torch.sum(torch.abs(layer_grad))
        return layer_grad_norm





