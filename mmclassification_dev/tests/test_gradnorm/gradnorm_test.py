import torch
import numpy as np
import matplotlib.pyplot as plt

outputs = torch.rand(10000, 1000)
logsoftmax = torch.nn.LogSoftmax(dim=-1)
loss = torch.mean(-logsoftmax(outputs), dim=-1).numpy()
out_softmax = torch.nn.functional.softmax(outputs, dim=1)
V = torch.norm((out_softmax), p=1, dim=1).numpy()
plt.scatter(loss, V)
plt.savefig("gradnorm_loss_gradient.pdf")
