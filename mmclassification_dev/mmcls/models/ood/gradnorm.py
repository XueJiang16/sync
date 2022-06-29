from mmcv.runner import BaseModule
import torch

from ..builder import OOD
from mmcls.models import build_classifier


@OOD.register_module()
class GradNorm(BaseModule):
    def __init__(self, classifier, num_classes, temperature, **kwargs):
        super(GradNorm, self).__init__()
        self.classifier = build_classifier(classifier)
        self.classifier.eval()
        self.num_classes = num_classes
        self.temperature = temperature
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
        print(kwargs)

    def forward(self, **input):
        self.classifier.zero_grad()
        img = input['img']
        assert img.shape[0] == 1, "GradNorm backward implementation only supports batch = 1."
        outputs = self.classifier.simple_test(softmax=False, **input)
        # outputs, _ = self.classifier.simple_test(softmax=False, **input)
        targets = torch.ones((img.shape[0], self.num_classes)).cuda()
        outputs = outputs / self.temperature
        loss = torch.sum(torch.mean(-targets * self.logsoftmax(outputs), dim=-1))

        loss.backward()
        layer_grad = self.classifier.head.layers[1].fc.weight.grad.data
        layer_grad_norm = torch.sum(torch.abs(layer_grad))
        return layer_grad_norm





