import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv import FileClient

# from .base_dataset import BaseDataset
from .builder import DATASETS
from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision

@DATASETS.register_module()
class TxtDataset(Dataset):
    def __init__(self, path, data_ann, transform, loader=None):
        super().__init__()
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_ann = data_ann
        self.loader = loader
        self.transform = transform
        with open(self.data_ann) as f:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]
        self.file_list = []
        self.label_list = []
        for filename, gt_label in samples:
            self.file_list.append(os.path.join(path, filename))
            self.label_list.append(int(gt_label))

    def __len__(self):
        return len(self.file_list)
        # return 32

    def __getitem__(self, item):
        path = self.file_list[item]
        sample = Image.open(path)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        label = self.label_list[item]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label
        # return sample, label, os.path.basename(path)

@DATASETS.register_module()
class JsonDataset(Dataset):
    # INaturalist
    def __init__(self, path, data_ann, transform, loader=None):
        super().__init__()
        # self.file_list = glob.glob(os.path.join(path, '*'))
        self.data_ann = data_ann
        self.loader = loader
        self.transform = transform
        with open(self.data_ann) as f:
            ann = json.load(f)
        images = ann['images']
        images_dict = dict()
        for item in images:
            images_dict[item['id']] = item['file_name']
        annotations = ann['annotations']
        samples = []
        for item in annotations:
            samples.append([images_dict[item['image_id']], item['category_id']])
        self.file_list = []
        self.label_list = []
        for filename, gt_label in samples:
            self.file_list.append(os.path.join(path, filename))
            self.label_list.append(int(gt_label))

    def __len__(self):
        return len(self.file_list)
        # return 32

    def __getitem__(self, item):
        path = self.file_list[item]
        sample = Image.open(path)
        if sample.mode != 'RGB':
            sample = sample.convert('RGB')
        label = self.label_list[item]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

@DATASETS.register_module()
class FolderDataset(torchvision.datasets.ImageFolder):
    pass
