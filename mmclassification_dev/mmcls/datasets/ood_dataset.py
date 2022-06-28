import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv import FileClient
from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision
import os
import copy

# from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose

class OODBaseDataset(Dataset):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = Compose(pipeline)
        self.file_list = []
        self.data_prefix = None

    def parse_datainfo(self):
        self.data_infos = []
        for sample in self.file_list:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': sample}
            self.data_infos.append(info)

    def __len__(self):
        return len(self.file_list)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.prepare_data(idx)


@DATASETS.register_module()
class TxtDataset(OODBaseDataset):
    def __init__(self, path, data_ann, pipeline):
        super().__init__(pipeline)
        self.data_prefix = path
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_ann = data_ann
        with open(self.data_ann) as f:
            samples = [x.strip().rsplit(' ', 1)[0] for x in f.readlines()]
        for filename in samples:
            self.file_list.append(filename)
        self.parse_datainfo()

@DATASETS.register_module()
class JsonDataset(OODBaseDataset):
    # INaturalist
    def __init__(self, path, data_ann, pipeline):
        super().__init__(pipeline)
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_prefix = path
        self.data_ann = data_ann
        with open(self.data_ann) as f:
            ann = json.load(f)
        images = ann['images']
        images_dict = dict()
        for item in images:
            images_dict[item['id']] = item['file_name']
        annotations = ann['annotations']
        samples = []
        for item in annotations:
            samples.append(images_dict[item['image_id']])
        for filename in samples:
            self.file_list.append(filename)
        self.parse_datainfo()


@DATASETS.register_module()
class FolderDataset(OODBaseDataset):
    def __init__(self, path, pipeline):
        super().__init__(pipeline)
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_prefix = path
        images = os.listdir(path)
        for filename in images:
            self.file_list.append(filename)
        self.parse_datainfo()