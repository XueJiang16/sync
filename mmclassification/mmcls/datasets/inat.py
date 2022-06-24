# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS
import json


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = os.path.join(root, folder_name)
        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples


@DATASETS.register_module()
class INaturalist(BaseDataset):
    """`INaturalist Dataset.
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = list(map(str, list(range(8142))))

    def load_annotations(self):
        if isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                ann = json.load(f)
            images = ann['images']
            images_dict = dict()
            for item in images:
                images_dict[item['id']] = item['file_name']
            annotations = ann['annotations']
            samples = []
            for item in annotations:
                samples.append([images_dict[item['image_id']], item['category_id']])
        else:
            raise TypeError('ann_file must be a str')
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
