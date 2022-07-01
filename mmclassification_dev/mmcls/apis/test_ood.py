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
        prog_bar = mmcv.ProgressBar(len(dataset))
    dist.barrier()
    for i, data in enumerate(data_loader):
        # print(data['img'].sum())
        # assert False
        result = model.forward(**data)
        if len(result.shape) == 0:  # handle the situation of batch = 1
            result = result.unsqueeze(0)
        results.append(result)
        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()
    dist.barrier()
    results = torch.cat(results).cpu().numpy()
    return results