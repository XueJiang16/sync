import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv import FileClient
from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision as tv
import os
import copy

# from .base_dataset import BaseDataset
from .builder import DATASETS
from .pipelines import Compose

class OODBaseDataset(Dataset):
    def __init__(self, name, pipeline):
        super().__init__()
        self.pipeline = Compose(pipeline)
        self.file_list = []
        self.data_prefix = None
        self.name = name
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize((480, 480)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([123.675/255, 116.28/255, 103.53/255],
                                    [58.395/255, 57.12/255, 57.375/255]),
        ])

    def parse_datainfo(self):
        self.data_infos = []
        for sample in self.file_list:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': sample}
            self.data_infos.append(info)

    def __len__(self):
        return 1024
        # return len(self.file_list)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        sample = Image.open(os.path.join(results['img_prefix'], results['img_info']['filename']))
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        results['img'] = sample
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.prepare_data(idx)


@DATASETS.register_module()
class TxtDataset(OODBaseDataset):
    def __init__(self, name, path, data_ann, pipeline):
        super().__init__(name, pipeline)
        self.data_prefix = path
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_ann = data_ann
        with open(self.data_ann) as f:
            samples = [x.strip().rsplit(' ', 1)[0] for x in f.readlines()]
        for filename in samples:
            self.file_list.append(filename)
        self.file_list.sort()
        self.parse_datainfo()

@DATASETS.register_module()
class JsonDataset(OODBaseDataset):
    # INaturalist
    def __init__(self, name, path, data_ann, pipeline):
        super().__init__(name, pipeline)
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
    def __init__(self, name, path, pipeline, data_ann=None):
        super().__init__(name, pipeline)
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_prefix = path
        images = os.listdir(path)
        for filename in images:
            self.file_list.append(filename)
        self.parse_datainfo()