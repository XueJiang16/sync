model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    # id_data=dict(
    #     type='TxtDataset',
    #     path='/data/csxjiang/val',
    #     data_ann='/data/csxjiang/meta/val_labeled.txt',
    #     pipeline=[
    #         dict(type='LoadImageFromFile'),
    #         dict(type='Resize', size=480),
    #         dict(
    #             type='Normalize',
    #             mean=[123.675, 116.28, 103.53],
    #             std=[58.395, 57.12, 57.375],
    #             to_rgb=True),
    #         dict(type='ImageToTensor', keys=['img']),
    #         dict(type='Collect', keys=['img'])
    #     ]),
    id_data=dict(
        type='JsonDataset',
        path='/data/csxjiang/',
        data_ann='/data/csxjiang/ood_data/inat/val2018.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=480),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    # ood_data=dict(
    #     type='TxtDataset',
    #     path='',
    #     data_ann='',
    #     transform=''
    # )
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
