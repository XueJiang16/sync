import numpy as np
import random
from collections import Counter


def main(a, num_img=128000, m=1):
    f = lambda x: a*(m**a) / x**(a+1)
    upper = 2.2
    lower = 1
    x = np.linspace(lower, upper, num=1000)
    y = f(x)
    scale_factor = num_img/sum(y)
    y = np.round(y * scale_factor)
    return y



if __name__ == '__main__':
    res = main(8)
    # obtain order
    # info = []
    # with open('/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled.txt', 'r') as f:
    #     ls = f.readlines()
    #     for l in ls:
    #         name, label = l.strip().split(' ')
    #         info.append(int(label))
    # label_stat = Counter(info)
    # order = sorted(label_stat.items(), key=lambda s: (-s[1]))
    # orders = []
    # for item in order:
    #     orders.append(item[0])
    # cls_num = [-1 for _ in range(1000)]
    # for i in range(1000):
    #     cat_idx = orders[i]
    #     cat_num = int(res[i])
    #     cls_num[cat_idx] = cat_num
    #     assert cat_num <= label_stat[cat_idx]
    #
    # print(cls_num)


    info = []
    with open('/mapai/haowenguo/data/ood_data/ImageNet-LT/ImageNet_LT_train.txt', 'r') as f:
        ls = f.readlines()
        for l in ls:
            name, label = l.strip().split(' ')
            info.append(int(label))
    label_stat = Counter(info)
    a = 3
    res = main(a=a,m=1)
    #obtain order
    info=[]
    with open('/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled.txt','r') as f:
        ls = f.readlines()
        for l in ls:
            name, label = l.strip().split(' ')
            info.append(int(label))
    label_stat = Counter(info)
    order = sorted(label_stat.items(), key=lambda s: (-s[1]))
    orders = []
    for item in order:
        orders.append(item[0])
    #sample
    info = dict()
    with open('/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_labeled.txt','r') as f:
        ls = f.readlines()
        random.shuffle(ls)
        for l in ls:
            name, label = l.strip().split(' ')
            if label not in info:
                info[label] = []
            info[label].append(name)
    for i in range(1000):
        cat_idx = orders[i]
        cat_num = int(res[i])
        info[str(cat_idx)] = info[str(cat_idx)][:cat_num]
    with open('/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a{}.txt'.format(a),'w') as f:
        for k, v in info.items():
            for img in v:
                f.write('{} {}\n'.format(img,k))
