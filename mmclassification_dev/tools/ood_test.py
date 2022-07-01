# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
from numbers import Number

import mmcv
import numpy as np
import torch
import time
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmcls.apis import single_gpu_test_ood
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_ood_model
from mmcls.utils import get_root_logger, setup_multi_processes, gather_tensors, evaluate_all


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
    cfg.gpu_ids = [int(os.environ['LOCAL_RANK'])]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if os.environ['LOCAL_RANK'] == '0':
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
        os.makedirs(cfg.work_dir, exist_ok=True)
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    dataset_id = build_dataset(cfg.data.id_data)
    dataset_ood = [build_dataset(d) for d in cfg.data.ood_data]
    name_ood = [d['name'] for d in cfg.data.ood_data]

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
    model.init_weights()
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    # model.to("cuda:{}".format(os.environ['LOCAL_RANK']))
    print()
    print("Processing in-distribution data...")
    outputs_id = single_gpu_test_ood(model, data_loader_id)
    in_scores = gather_tensors(outputs_id)
    in_scores = np.concatenate(in_scores, axis=0)

    # out_scores_list = []
    for ood_set, ood_name in zip(data_loader_ood, name_ood):
        print()
        print("Processing out-of-distribution data ({})...".format(ood_name))
        outputs_ood = single_gpu_test_ood(model, ood_set)
        out_scores = gather_tensors(outputs_ood)
        out_scores = np.concatenate(out_scores, axis=0)
        # out_scores_list.append(out_scores)
        if os.environ['LOCAL_RANK'] == '0':
            auroc, aupr_in, aupr_out, fpr95 = evaluate_all(in_scores, out_scores)
            logger.info('============Overall Results for {}============'.format(ood_name))
            logger.info('AUROC: {}'.format(auroc))
            logger.info('AUPR (In): {}'.format(aupr_in))
            logger.info('AUPR (Out): {}'.format(aupr_out))
            logger.info('FPR95: {}'.format(fpr95))
            logger.info('quick data: {},{},{},{}'.format(auroc, aupr_in, aupr_out, fpr95))


if __name__ == '__main__':
    main()
