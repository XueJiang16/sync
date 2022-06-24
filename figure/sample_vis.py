import numpy as np
import random
from collections import Counter

def main(a,m=1, num_img=128000):
    f = lambda x: a*(m**a) / x**(a+1)
    upper = 2.2
    lower = 1
    x = np.linspace(lower, upper, num=1000)
    y = f(x)
    scale_factor = num_img/sum(y)
    y = np.round(y * scale_factor)
    return y

import matplotlib.pyplot as plt

plt.figure()
for i in range(2,9):
    y = main(i)
    # print(y.max())
    sample_name='ImageNet-LT-a{}'.format(i)
    # print(sample_name)
    plt.plot(y,label=sample_name)
    plt.legend(loc='best', fontsize=14)
    plt.xlabel('Sample Class', fontsize=18, fontweight='bold')
    plt.ylabel('Sample Number', fontsize=18, fontweight='bold')
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.tight_layout()
plt.show()
plt.savefig('sample.pdf')
exit()
y = main(8)
ba_imagenet = '/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled_10percent.txt'
with open(ba_imagenet, 'r') as f:
    lines = f.readlines()
y_ = []
for line in lines:
    class_name = int(line.split()[-1].strip())
    y_.append(class_name)
res = Counter(y_)
y__=[]
for k in range(1000):
    y__.append(res[k])
# y_ = [128]*1000
# print(y.max())
sample_name='ImageNet-LT-a8'
# print(sample_name)
plt.plot(y,label=sample_name)
plt.plot(y__, label='ImageNet-Balanced')
plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.16), ncol=2, fontsize=14)
plt.xlabel('Sample Class', fontsize=24, fontweight='bold')
plt.ylabel('Sample Number', fontsize=24, fontweight='bold')
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()
plt.show()
plt.savefig('fig1.pdf')