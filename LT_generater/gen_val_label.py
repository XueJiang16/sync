import numpy as np
import random
from collections import Counter

if __name__ == '__main__':
    for a in range(2,9):
        info = []
        with open('/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a{}.txt'.format(a),'r') as f:
            ls = f.readlines()
            for l in ls:
                name, label = l.strip().split(' ')
                info.append(int(label))
        label_stat = Counter(info)
        label_cls = dict()
        for k, v in label_stat.items():
            # if v > 100:
            #     label_cls[k]=2 #head
            # elif v < 20:
            #     label_cls[k]=0 #tail
            # else:
            #     label_cls[k]=1 #mid
            label_cls[k]=v
        info = dict()
        with open('/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/val_labeled.txt','r') as f:
            ls = f.readlines()
            # random.shuffle(ls)
            for l in ls:
                name, label = l.strip().split(' ')
                cls = label_cls[int(label)]
                if cls not in info:
                    info[cls] = []
                info[cls].append(name)

        with open('/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/val_LT_a{}_counter.txt'.format(a),'w') as f:
            for k, v in info.items():
                for img in v:
                    f.write('{} {}\n'.format(img, k))
