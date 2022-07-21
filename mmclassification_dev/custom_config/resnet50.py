method_name = 'Energy'
model_name = 'resnet50'
train_dataset = 'Balance'
custom_name = None
if custom_name is not None:
    readable_name = '{}_{}_{}_{}'.format(method_name, model_name, train_dataset, custom_name)
else:
    readable_name ='{}_{}_{}'.format(method_name, model_name, train_dataset)
quick_test = True
training_file = None
# training_file = '/data/csxjiang/meta/train_LT_a8.txt'
model = dict(
    type=method_name,
    debug_mode=False,
    num_classes=1000,
    # temperature=1,
    target_file=training_file,
    classifier=dict(
        type='ImageClassifier',
        init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/pytorch_official/resnet50_custom.pth'),
        # init_cfg=dict(type='Pretrained', checkpoint='/data/csxjiang/ood_ckpt/ckpt/resnet50_LT_a8/epoch_100.pth'),
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            style='pytorch',
            random_block=1,
            random_block_k=2.5,
        ),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5))
    )
)
# pipline =[dict(type='Collect', keys=['img'])]
pipline =[dict(type='Collect', keys=['img', 'type'])]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    id_data=dict(
        name='ImageNet',
        type='TxtDataset',
        path='/data/csxjiang/val',
        data_ann='/data/csxjiang/meta/val_labeled.txt',
        pipeline=pipline,
        len_limit=5000 if quick_test else -1,
        train_label=training_file,
    ),
    # id_data=dict(
    #     type='JsonDataset',
    #     path='/data/csxjiang/',
    #     data_ann='/data/csxjiang/ood_data/inat/val2018.json',
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
    ood_data=[
        dict(
            name='iNaturalist',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/iNaturalist/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='SUN',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/SUN/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Places',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Places/images',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
        dict(
            name='Textures',
            type='FolderDataset',
            path='/data/csxjiang/ood_data/Textures/dtd/images_collate',
            pipeline=pipline,
            len_limit=1000 if quick_test else -1,
        ),
    ],

)
dist_params = dict(backend='nccl')
log_level = 'CRITICAL'
# log_level = 'INFO'
work_dir = './results/'
