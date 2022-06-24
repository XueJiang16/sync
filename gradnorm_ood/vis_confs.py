import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter
import tqdm
from concurrent_helper import run_with_concurrent

def main(args):
    in_file = np.load(args.in_confs)
    in_confs = in_file['in_confs']
    in_labels = in_file['labels']
    # in_confs_argmax = in_confs.argmax(axis=-1)
    # hit = np.equal(in_confs_argmax, in_labels)
    # print(np.sum(hit) / len(in_confs))
    # exit()
    out_file = np.load(args.out_confs)
    out_confs = out_file['out_confs']
    label_filename = args.id_cls
    cls_idx = []
    with open(label_filename, 'r') as f:   #train set
        for line in f.readlines():
            segs = line.strip().split(' ')
            cls_idx.append(int(segs[-1]))
    cls_idx = np.array(cls_idx, dtype='int')
    res = Counter(cls_idx)
    cls_num = []
    for k,v in res.items():
        cls_num.append((k,v))
    cls_num.sort(key=lambda x: x[1], reverse=True)
    cls_rank = []
    for k, v in cls_num:
        cls_rank.append(k)
    #each avg
    # sorted_label = in_labels.argsort()
    #     # in_confs = in_confs[sorted_label].reshape((50,-1, 1000))
    #     # in_example = np.mean(in_confs, axis=0)
    for i in tqdm.trange(1000):
        confs = in_confs[in_labels==i]
        confs = np.mean(confs, axis=0)
        freq = res[i]
        plt.figure(figsize=(10,10))
        plt.bar(range(len(confs)), confs)
        plt.savefig('./files/in_each_confs_a4/{}_cls_freq_{}.jpg'.format(i, freq))
        plt.close()


    #avg
    # in_example = np.mean(in_confs, axis=0)
    # in_example = in_example[cls_rank]   #类别从多到少排序1000维
    # plt.figure(1)
    # plt.bar(range(len(in_example)), in_example)
    # plt.savefig('./files/in_confs_a4.jpg')
    # out_example = np.mean(out_confs, axis=0)
    # out_example = out_example[cls_rank]
    # plt.figure(2)
    # plt.bar(range(len(out_example)), out_example)
    # plt.savefig('./files/out_confs_a4_inature.jpg')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_confs", default='files/id_confs_a4.npz',help="Path to the in-distribution data folder.")
    parser.add_argument("--out_confs", default='files/ood_confs_a4_inature.npz',help="Path to the out-of-distribution data folder.")
    parser.add_argument("--id_cls", default='/mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/train_LT_a4.txt', help="id label file")

    args = parser.parse_args()
    main(args)
