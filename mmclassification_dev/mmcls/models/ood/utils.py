import torch
import numpy as np
import os
import torch.nn.functional as F


def print_category(outputs, softmax=False):
    pred = outputs.detach().clone()
    if softmax:
        pred = F.softmax(pred, dim=1)
    if os.environ['LOCAL_RANK'] == '0':
        print(torch.argmax(pred, dim=1))


