from mmcls.apis import multi_gpu_test, single_gpu_test
# from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
# from mmcls.utils import get_root_logger, setup_multi_processes
from mmcv.runner import load_checkpoint
import torch

model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV3OOD', arch='large'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='OODHead',
        num_classes=1000,
        in_channels=960,
        mid_channels=[1280],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))

ckpt = '/mapai/haowenguo/ckpt/ood_ckpt/ckpt/mobile_LT_a8/epoch_600.pth'
model = build_classifier(model)
checkpoint = load_checkpoint(model, ckpt)
print(list(model.named_parameters()))
#
# data = torch.randn((1,3,100,100), dtype=torch.float32, device='cuda')
# model = model.cuda()
# result = model(return_loss=False, img=data, img_metas=None)

