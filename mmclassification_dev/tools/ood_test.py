# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmcls.apis import single_gpu_test_ood
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_ood_model
from mmcls.utils import get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='ood test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'ipu'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    #
    # if args.gpu_ids is not None:
    #     cfg.gpu_ids = args.gpu_ids[0:1]
    #     warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
    #                   'Because we only support single GPU mode in '
    #                   'non-distributed testing. Use the first GPU '
    #                   'in `gpu_ids` now.')
    # else:
    cfg.gpu_ids = [os.environ['LOCAL_RANK']]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset_id = build_dataset(cfg.data.id_data)
    dataset_ood = [build_dataset(d) for d in cfg.data.ood_data]

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if args.device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'id_data', 'ood_data'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader_id = build_dataloader(dataset_id, **test_loader_cfg)
    data_loader_ood = []
    for ood_set in dataset_ood:
        data_loader_ood.append(build_dataloader(ood_set, **test_loader_cfg))

    model = build_ood_model(cfg.model)
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    # model.to("cuda:{}".format(os.environ['LOCAL_RANK']))
    outputs_id = single_gpu_test_ood(model, data_loader_id)



    # for i, data in enumerate(data_loader):
    #     print(data)
    #     assert False

    # build the model and load checkpoint
    # model = build_classifier(cfg.model)
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    #
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     from mmcls.datasets import ImageNet
    #     warnings.simplefilter('once')
    #     warnings.warn('Class names are not saved in the checkpoint\'s '
    #                   'meta data, use imagenet by default.')
    #     CLASSES = ImageNet.CLASSES
    #
    # if not distributed:
    #     if args.device == 'cpu':
    #         model = model.cpu()
    #     else:
    #         model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    #         if not model.device_ids:
    #             assert mmcv.digit_version(mmcv.__version__) >= (1, 4, 4), \
    #                 'To test with CPU, please confirm your mmcv version ' \
    #                 'is not lower than v1.4.4'
    #     model.CLASSES = CLASSES
    #     show_kwargs = {} if args.show_options is None else args.show_options
    #     outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
    #                               **show_kwargs)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)
    #
    # rank, _ = get_dist_info()
    # if rank == 0:
    #     results = {}
    #     logger = get_root_logger()
    #     if args.metrics:
    #         eval_results = dataset.evaluate(
    #             results=outputs,
    #             metric=args.metrics,
    #             metric_options=args.metric_options,
    #             logger=logger)
    #         results.update(eval_results)
    #         for k, v in eval_results.items():
    #             if isinstance(v, np.ndarray):
    #                 v = [round(out, 2) for out in v.tolist()]
    #             elif isinstance(v, Number):
    #                 v = round(v, 2)
    #             else:
    #                 raise ValueError(f'Unsupport metric type: {type(v)}')
    #             print(f'\n{k} : {v}')
    #     if args.out:
    #         if 'none' not in args.out_items:
    #             scores = np.vstack(outputs)
    #             pred_score = np.max(scores, axis=1)
    #             pred_label = np.argmax(scores, axis=1)
    #             pred_class = [CLASSES[lb] for lb in pred_label]
    #             res_items = {
    #                 'class_scores': scores,
    #                 'pred_score': pred_score,
    #                 'pred_label': pred_label,
    #                 'pred_class': pred_class
    #             }
    #             if 'all' in args.out_items:
    #                 results.update(res_items)
    #             else:
    #                 for key in args.out_items:
    #                     results[key] = res_items[key]
    #         print(f'\ndumping results to {args.out}')
    #         mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()
