import torch
import numpy as np
import matplotlib.pyplot as plt

outputs = torch.rand(10000, 1000)
logsoftmax = torch.nn.LogSoftmax(dim=-1)
target = torch.ones((10000, 1000))
loss = torch.mean(-target * logsoftmax(outputs), dim=-1).numpy()
out_softmax = torch.nn.functional.softmax(outputs, dim=1)
V = torch.norm((target - out_softmax), p=1, dim=1).numpy()
plt.plot(loss, V)
plt.savefig("gradnorm_loss_gradient.pdf")
