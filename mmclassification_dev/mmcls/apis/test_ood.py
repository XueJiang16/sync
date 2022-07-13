import os.path as osp
import pickle
import random
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
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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
        data['dataset_name'] = name
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

def ssim_test(img, img_metas=None, **kwargs):
    crop_size = 120
    img_size = 480
    crops = []
    crops_mean = []
    crops_std = []
    img = img[0].permute(1,2,0).contiguous().cpu().numpy()
    # for i in range(10):
    #     crop_x = random.randint(0, 480-crop_size)
    #     crop_y = random.randint(0, 480-crop_size)
    #     crop = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size, :]
    #     crops.append(crop)
    ssim_crops = 0
    # for i in range(0,10,2):
    #     # psnr_temp = psnr(crops[i], crops[i+1], data_range=img.max() - img.min())
    #     # ssim_crops += psnr_temp if not np.isinf(psnr_temp) else 100
    #     # ssim_crops += ssim(crops[i], crops[i+1], data_range=img.max() - img.min(), channel_axis=2)
    #     # mean_bias = np.abs(crops[i].mean(axis=2) - crops[i+1].mean(axis=2)).sum()
    #     mean_bias = np.abs(crops[i].mean() - crops[i+1].mean())
    #     std_bias = np.abs(crops[i].std() - crops[i+1].std())
    #     ssim_crops += (mean_bias + 3*std_bias)
    #     # ssim_crops += std_bias
    # ssim_crops /= 5
    corner_list = []
    for h in range(img_size//crop_size):
        for w in range(img_size//crop_size):
            corner_list.append([h*crop_size, w*crop_size])
    for h,w in corner_list:
        crop = img[h:h+crop_size, w:w+crop_size, :]
        crops_mean.append(crop.mean())
        crops_std.append(crop.std())
    # percentile
    crops_mean.sort()
    crops_std.sort()
    num_crops = len(crops_mean)
    lower_percentile = 0.1
    upper_percentile = 0.9
    lower_bound = int(lower_percentile * num_crops)
    upper_bound = int(upper_percentile * num_crops)
    crops_mean = crops_mean[lower_bound: upper_bound]
    crops_std = crops_std[lower_bound: upper_bound]
    ssim_crops = np.std(crops_mean) + 3 * np.std(crops_std)
    return ssim_crops


def single_gpu_test_ssim(model,
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
        data['dataset_name'] = name
        result = ssim_test(**data)
        result = torch.tensor(result).to('cuda:{}'.format(rank))
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
    # if rank == 0:
    #     x = np.arange(1, 1001, 1)
    #     cat_scores = torch.cat(cat_scores).mean(dim=0).cpu().numpy()
    #     plt.figure(figsize=(20, 20))
    #     plt.plot(x, cat_scores)
    #     plt.savefig("{}_score.pdf".format(name))
    #     plt.close()
    results = torch.cat(results).cpu().numpy()
    return results