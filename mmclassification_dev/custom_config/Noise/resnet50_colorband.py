method_name = 'GradNormBatch'
model_name = 'resnet50'
train_dataset = 'Balance'
custom_name = "NoiseColorBand"
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=1000,
    temperature=1,
    # target_file='/data/csxjiang/meta/train_LT_a8.txt',
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/pytorch_official/resnet50_custom.pth'),
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))
    )
)
pipline =[
    dict(type='Collect', keys=['img', 'type'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    id_data=dict(
        name='ImageNet',
        type='TxtDataset',
        path='/data/csxjiang/val',
        data_ann='/data/csxjiang/meta/val_labeled.txt',
        pipeline=pipline),
    ood_data=[
        dict(
            name='NoiseColorBand',
            type='NoiseDatasetColorBand',
            length=10000,
            img_size=224,
            pipeline=pipline),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/'
