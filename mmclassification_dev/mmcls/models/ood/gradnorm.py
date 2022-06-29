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

    def simple_test(self, img, img_metas=None, **kwargs):
        self.classifier.zero_grad()
        outputs, _ = self.classifier(img)
        targets = torch.ones((img.shape[0], self.num_classes)).cuda()
        outputs = outputs / self.temperature
        loss = torch.sum(torch.mean(-targets * self.logsoftmax(outputs), dim=-1))

        loss.backward()
        layer_grad = self.classifier.head.layers[1].fc.weight.grad.data
        layer_grad_norm = torch.sum(torch.abs(layer_grad))
        return layer_grad_norm





