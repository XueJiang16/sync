import torch
import numpy as np
import matplotlib.pyplot as plt

criterion = torch.nn.CrossEntropyLoss()
outputs = torch.rand(10000, 1000)
maxIndexTemp_numpy = np.argmax(outputs.numpy(), axis=1)
maxIndexTemp_torch = torch.argmax(outputs, dim=1)
bias = np.allclose(maxIndexTemp_torch.numpy(), maxIndexTemp_numpy)
# print(bias)
assert bias
# temperature = 1000
# outputs = outputs / temperature
# labels = torch.LongTensor(maxIndexTemp).cuda()
# loss = criterion(outputs, labels)
# assert False


# Calculating the confidence after adding perturbations
nnOutputs = outputs

nnOutputs_numpy = nnOutputs.numpy()
nnOutputs_numpy = nnOutputs_numpy - np.max(nnOutputs_numpy, axis=1, keepdims=True)
nnOutputs_numpy = np.exp(nnOutputs_numpy) / np.sum(np.exp(nnOutputs_numpy), axis=1, keepdims=True)
confs = np.max(nnOutputs_numpy, axis=1)
# confs = torch.tensor(confs)
nnOutputs_torch = nnOutputs - torch.max(nnOutputs, dim=1, keepdim=True)[0]
nnOutputs_torch = torch.exp(nnOutputs_torch) / torch.sum(torch.exp(nnOutputs_torch), dim=1, keepdim=True)
confs_torch, _ = torch.max(nnOutputs_torch, dim=1)
bias2 = np.allclose(confs, confs_torch.numpy())
assert bias2

