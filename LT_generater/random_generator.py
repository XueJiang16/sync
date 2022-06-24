import numpy as np
import random
from collections import Counter


def sample(a, num_img=128000, m=1):
    f = lambda x: a*(m**a) / x**(a+1)
    upper = 2.2
    lower = 1
    x = np.linspace(lower, upper, num=1000)
    y = f(x)
    scale_factor = num_img/sum(y)
    y = np.round(y * scale_factor)
    return y

def generate(a, idx):
    res = sample(a=a, m=1)
    # obtain order
    info = []
    index = []
    with open('/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled.txt', 'r') as f:
        ls = f.readlines()
        for l in ls:
            name, label = l.strip().split(' ')
            info.append(int(label))
    label_stat = Counter(info)
    for k,v in label_stat.items():
        index.append((k, v))
    random.shuffle(index)

    # order = sorted(label_stat.items(), key=lambda s: (-s[1]))
    # orders = []
    # for item in order:
    #     orders.append(item[0])
    # sample
    info = dict()
    with open('/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled.txt', 'r') as f:
        ls = f.readlines()
        random.shuffle(ls)
        for l in ls:
            name, label = l.strip().split(' ')
            if label not in info:
                info[label] = []
            info[label].append(name)  #sample source
    for i in range(1000):
        cat_num = int(res[i])
        for j in range(len(index)):
            if index[j][1] > cat_num:
                cat_idx = index[j][0]
                info[str(cat_idx)] = info[str(cat_idx)][:cat_num]
                index.pop(j)
                break
    with open('/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_repeat{}_a{}.txt'.format(idx, a), 'w') as f:
        for k, v in info.items():
            for img in v:
                f.write('{} {}\n'.format(img, k))


if __name__ == '__main__':
    # for i in range(7,9):
    i=2
    for j in range(1,11):
        generate(i, j)
