import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter
import tqdm
import seaborn as sns
import os

def single_picture(basic_path, method, dataset, mode=''):
    path = os.path.join(basic_path, method, dataset)
    if mode == 'rw':
        in_confs = np.load(path+'/id_rw.npy')
        in_confs = in_confs[:,0]
        head_confs = np.load(path+'/head_rw.npy')
        head_confs = head_confs[:,0]
        mid_confs = np.load(path+'/mid_rw.npy')
        mid_confs = mid_confs[:,0]
        tail_confs = np.load(path+'/tail_rw.npy')
        tail_confs = tail_confs[:,0]
        out_confs = np.load(path+'/ood_rw.npy')
        out_confs = out_confs[:,0]
        #id-ood
        sns.set(rc={'figure.figsize': (8, 6)})
        # sns.set_style('whitegrid')
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        fig = sns.kdeplot(np.array(in_confs), bw=0.2)
        sns.kdeplot(np.array(out_confs), bw=0.2)
        plt.legend(labels=['ID (ImageNet)', 'OOD ({})'.format(dataset)])
        plt.xlabel('OOD scores', fontsize=18, fontweight='bold')
        plt.ylabel('Density', fontsize=18, fontweight='bold')
        plt.yticks(size=16)
        plt.xticks(size=16)
        plt.tight_layout()
        fig.get_figure().savefig('dis_file/rw_{}_{}_overall.pdf'.format(method, dataset))
        plt.close()

        #tail
        sns.set(rc={'figure.figsize': (8, 6)})
        sns.set_style('whitegrid')
        fig = sns.kdeplot(np.array(head_confs), bw=0.2)
        sns.kdeplot(np.array(mid_confs), bw=0.2)
        sns.kdeplot(np.array(tail_confs), bw=0.2)
        sns.kdeplot(np.array(out_confs), bw=0.2)
        plt.legend(labels=['ID-Head', 'ID-Mid', 'ID-Tail', 'OOD ({})'.format(dataset)])
        plt.xlabel('OOD scores', fontsize=18, fontweight='bold')
        plt.ylabel('Density', fontsize=18, fontweight='bold')
        plt.yticks(size=16)
        plt.xticks(size=16)
        fig.get_figure().savefig('dis_file/rw_{}_{}.pdf'.format(method, dataset))
        plt.tight_layout()
        plt.close()
    else:
        in_confs = np.load(path+'/id.npy')
        in_confs = in_confs[:,0]
        head_confs = np.load(path+'/head.npy')
        head_confs = head_confs[:,0]
        mid_confs = np.load(path+'/mid.npy')
        mid_confs = mid_confs[:,0]
        tail_confs = np.load(path+'/tail.npy')
        tail_confs = tail_confs[:,0]
        out_confs = np.load(path+'/ood.npy')
        out_confs = out_confs[:,0]
        #id-ood
        sns.set(rc={'figure.figsize': (8, 6)})
        # sns.set_style('whitegrid')
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        fig = sns.kdeplot(np.array(in_confs), bw=0.2)
        sns.kdeplot(np.array(out_confs), bw=0.2)
        plt.legend(labels=['ID (ImageNet)', 'OOD ({})'.format(dataset)])
        plt.xlabel('OOD scores', fontsize=18, fontweight='bold')
        plt.ylabel('Density', fontsize=18, fontweight='bold')
        plt.yticks(size=16)
        plt.xticks(size=16)
        fig.get_figure().savefig('dis_file/{}_{}_overall.pdf'.format(method, dataset))
        plt.close()

        #tail
        sns.set(rc={'figure.figsize': (8, 6)})
        sns.set_style('whitegrid')
        fig = sns.kdeplot(np.array(head_confs), bw=0.2)
        sns.kdeplot(np.array(mid_confs), bw=0.2)
        sns.kdeplot(np.array(tail_confs), bw=0.2)
        sns.kdeplot(np.array(out_confs), bw=0.2)
        plt.legend(labels=['ID-Head', 'ID-Mid', 'ID-Tail', 'OOD ({})'.format(dataset)])
        plt.xlabel('OOD scores', fontsize=18, fontweight='bold')
        plt.ylabel('Density', fontsize=18, fontweight='bold')
        plt.yticks(size=16)
        plt.xticks(size=16)
        fig.get_figure().savefig('dis_file/{}_{}.pdf'.format(method, dataset))
        plt.close()

def subplots(basic_path, method, datasets, mode=''):
    path_o = os.path.join(basic_path, method)
    if mode == 'rw':
        in_confs = []
        head_confs = []
        mid_confs = []
        tail_confs = []
        out_confs = []
        for i in range(len(datasets)):
            path = os.path.join(path_o, datasets[i])
            in_conf = np.load(path + '/id_rw.npy')
            in_conf = in_conf[:, 0]
            in_confs.append(in_conf)
            head_conf = np.load(path + '/head_rw.npy')
            head_conf = head_conf[:, 0]
            head_confs.append(head_conf)
            mid_conf = np.load(path + '/mid_rw.npy')
            mid_conf = mid_conf[:, 0]
            mid_confs.append(mid_conf)
            tail_conf = np.load(path + '/tail_rw.npy')
            tail_conf = tail_conf[:, 0]
            tail_confs.append(tail_conf)
            out_conf = np.load(path + '/ood_rw.npy')
            out_conf = out_conf[:, 0]
            out_confs.append(out_conf)
        # id-ood
        # figs, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(32, 7))
        # for i, ax in enumerate(axes.flatten()):
        #     sns.set(rc={'figure.figsize': (8, 6)})
        #     sns.set_style('whitegrid')
        #     fig = sns.kdeplot(np.array(in_confs[i]), bw=0.2, ax=ax)
        #     sns.kdeplot(np.array(out_confs[i]), bw=0.2, ax=ax)
        #     ax.legend(labels=['ID (ImageNet)', 'OOD ({})'.format(datasets[i])], fontsize=16)
        #     ax.set_xlabel('OOD scores', size=18, fontweight='bold')
        #     ax.set_ylabel('Density', size=18, fontweight='bold')
        # plt.tight_layout()
        # # fig.get_figure().savefig('dis_file/concat_rw_{}_overall.pdf'.format(method))
        # figs.savefig('dis_file/concat_rw_{}_overall.pdf'.format(method))
        # plt.close()

        # tail
        figs, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(32, 7))
        for i, ax in enumerate(axes.flatten()):
            sns.set(rc={'figure.figsize': (8, 6)})
            sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
            fig = sns.kdeplot(np.array(in_confs[i]), bw=0.2, ax=ax)
            sns.kdeplot(np.array(head_confs[i]), bw=0.2, ax=ax)
            sns.kdeplot(np.array(mid_confs[i]), bw=0.2, ax=ax)
            sns.kdeplot(np.array(tail_confs[i]), bw=0.2, ax=ax)
            sns.kdeplot(np.array(out_confs[i]), bw=0.2, ax=ax)
            ax.legend(labels=['ID (ImageNet)','ID-Head', 'ID-Mid', 'ID-Tail', 'OOD ({})'.format(datasets[i])], fontsize=24)
            ax.set_xlabel('OOD Scores', size=28, fontweight='bold')
            ax.set_ylabel('Density', size=28, fontweight='bold')
        plt.tight_layout()
        figs.savefig('dis_file/concat_rw_{}.pdf'.format(method))
        plt.close()
    else:
        in_confs = []
        head_confs = []
        mid_confs = []
        tail_confs = []
        out_confs = []
        for i in range(len(datasets)):
            path = os.path.join(path_o, datasets[i])
            in_conf = np.load(path + '/id.npy')
            in_conf = in_conf[:, 0]
            in_confs.append(in_conf)
            head_conf = np.load(path + '/head.npy')
            head_conf = head_conf[:, 0]
            head_confs.append(head_conf)
            mid_conf = np.load(path + '/mid.npy')
            mid_conf = mid_conf[:, 0]
            mid_confs.append(mid_conf)
            tail_conf = np.load(path + '/tail.npy')
            tail_conf = tail_conf[:, 0]
            tail_confs.append(tail_conf)
            out_conf = np.load(path + '/ood.npy')
            out_conf = out_conf[:, 0]
            out_confs.append(out_conf)
        # id-ood
        # figs, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(32, 7))
        # for i, ax in enumerate(axes.flatten()):
        #     sns.set(rc={'figure.figsize': (8, 6)})
        #     sns.set_style('whitegrid')
        #     fig = sns.kdeplot(np.array(in_confs[i]), bw=0.2, ax=ax)
        #     sns.kdeplot(np.array(out_confs[i]), bw=0.2, ax=ax)
        #     ax.legend(labels=['ID (ImageNet)', 'OOD ({})'.format(datasets[i])], fontsize=16)
        #     ax.set_xlabel('OOD scores', size=18, fontweight='bold')
        #     ax.set_ylabel('Density', size=18, fontweight='bold')
        # plt.tight_layout()
        # figs.savefig('dis_file/concat_{}_overall.pdf'.format(method))
        # plt.close()

        # tail
        figs, axes = plt.subplots(nrows=1, ncols=4, figsize=(32, 7))
        for i, ax in enumerate(axes.flatten()):
            sns.set(rc={'figure.figsize': (8, 6)})
            sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
            fig = sns.kdeplot(np.array(in_confs[i]), bw=0.2, ax=ax)
            sns.kdeplot(np.array(head_confs[i]), bw=0.2, ax=ax)
            sns.kdeplot(np.array(mid_confs[i]), bw=0.2, ax=ax)
            sns.kdeplot(np.array(tail_confs[i]), bw=0.2, ax=ax)
            sns.kdeplot(np.array(out_confs[i]), bw=0.2, ax=ax)
            ax.legend(labels=['ID (ImageNet)','ID-Head', 'ID-Mid', 'ID-Tail', 'OOD ({})'.format(datasets[i])], fontsize=24)
            ax.set_xlabel('OOD Scores', size=28, fontweight='bold')
            ax.set_ylabel('Density', size=28, fontweight='bold')
        plt.tight_layout()
        figs.savefig('dis_file/concat_{}.pdf'.format(method))
        plt.close()


if __name__ == '__main__':
    basic_path = '/mapai/haowenguo/code/SPL/jx/gradnorm_ood/dis_file/others'
    # methods = ['MSP','ODIN','Energy']
    # methods = ['GradNorm']
    methods = ['new']
    datasets = ['iNaturalist','SUN','Places','Textures']
    for method in methods:
        # subplots(basic_path, method, datasets, mode='rw')
        subplots(basic_path, method, datasets, mode='')
        subplots(basic_path, method, datasets, mode='')
        subplots(basic_path, method, datasets, mode='rw')
        # for dataset in datasets:
        #     single_picture(basic_path, method, dataset, mode='rw')
