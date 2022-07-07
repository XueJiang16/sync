import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import matplotlib.pyplot as plt


def single_gpu_test_ood(model,
                        data_loader,
                        name=''
                        ):
    """Test model with local single gpu.

    This method tests model with a single gpu and supports showing results.

    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.
        show (bool): Whether to show the test results. Defaults to False.
        out_dir (str): The output directory of result plots of all samples.
            Defaults to None, which means not to write output files.
        **show_kwargs: Any other keyword arguments for showing results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog = 0
        tic = time.time()
        # prog_bar = mmcv.ProgressBar(len(dataset))
    if world_size > 1:
        dist.barrier()
    for i, data in enumerate(data_loader):
        result = model.forward(**data)
        if len(result.shape) == 0:  # handle the situation of batch = 1
            result = result.unsqueeze(0)
        results.append(result)
        if rank == 0:
            batch_size = data['img'].size(0)
            prog += batch_size * world_size
            toc = time.time()
            passed_time = toc - tic
            inf_speed = passed_time / prog
            fps = 1 / inf_speed
            eta = max(0, (len(dataset) - prog)) * inf_speed
            print("[{} @ {}] {} / {}, fps = {}, eta = {}"
                  .format(name, int(passed_time), prog, len(dataset), round(fps, 2), round(eta, 2)))
    if world_size > 1:
        dist.barrier()
    results = torch.cat(results).cpu().numpy()
    return results

def single_gpu_test_ood_score(model,
                              data_loader,
                              name=''
                              ):
    """Test model with local single gpu.

    This method tests model with a single gpu and supports showing results.

    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.
        show (bool): Whether to show the test results. Defaults to False.
        out_dir (str): The output directory of result plots of all samples.
            Defaults to None, which means not to write output files.
        **show_kwargs: Any other keyword arguments for showing results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        cat_scores = []
        prog = 0
        tic = time.time()
        # prog_bar = mmcv.ProgressBar(len(dataset))
    if world_size > 1:
        dist.barrier()
    for i, data in enumerate(data_loader):
        result, cat_score = model.forward(**data)
        if len(result.shape) == 0:  # handle the situation of batch = 1
            result = result.unsqueeze(0)
        results.append(result)
        if rank == 0:
            cat_scores.append(cat_score)
            batch_size = data['img'].size(0)
            prog += batch_size * world_size
            toc = time.time()
            passed_time = toc - tic
            inf_speed = passed_time / prog
            fps = 1 / inf_speed
            eta = max(0, (len(dataset) - prog)) * inf_speed
            print("[{} @ {}] {} / {}, fps = {}, eta = {}"
                  .format(name, int(passed_time), prog, len(dataset), round(fps, 2), round(eta, 2)))
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        x = np.arange(1, 1001, 1)
        cat_scores = torch.cat(cat_scores).mean(dim=0).cpu().numpy()
        plt.figure(figsize=(20, 20))
        plt.plot(x, cat_scores)
        plt.savefig("{}_score.pdf".format(name))
        plt.close()
    results = torch.cat(results).cpu().numpy()
    return results