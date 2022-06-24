import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as tick



# def vis_single(data_list, name, mode='AUROC'):


if __name__ == '__main__':
    data1 = '''82.51	72.19   71.26	80.45
82.68	66.48	69.07	76.48
83.85	69.50	71.91	78.77
82.51	75.77	71.97	83.01'''
    """gradnorm:head,mid,tail"""
    data1 = data1.split('\n')
    data_list1 = [list(map(float, x.split())) for x in data1]

    data2 = '''91.23	43.83   75.92	70.43
87.56	51.94	67.49	75.53
91.42	42.26	75.93	69.72
93.38	37.58	80.78	66.47'''

    # imagenet-a8
    data2 = data2.split('\n')
    data_list2 = [list(map(float, x.split())) for x in data2]
    # vis_single([head1, mid1, tail1,head2, mid2, tail2],'GradNorm','AUROC')
    datasets = ['iNaturalist', 'Average']
    metrics = ['AUROC', 'FPR95']
    y = ['Overall','Head', 'Mid', 'Tail']
    all1_fpr = data_list1[0][1::2]
    head1_fpr = data_list1[1][1::2]
    mid1_fpr = data_list1[2][1::2]
    tail1_fpr = data_list1[3][1::2]
    all2_fpr = data_list2[0][1::2]
    head2_fpr = data_list2[1][1::2]
    mid2_fpr = data_list2[2][1::2]
    tail2_fpr = data_list2[3][1::2]

    all1_auroc = data_list1[0][::2]
    head1_auroc = data_list1[1][::2]
    mid1_auroc = data_list1[2][::2]
    tail1_auroc = data_list1[3][::2]
    all2_auroc = data_list2[0][::2]
    head2_auroc = data_list2[1][::2]
    mid2_auroc = data_list2[2][::2]
    tail2_auroc = data_list2[3][::2]

    x1 = []
    x2 = []
    for i in range(len(datasets)):
        x1.append([all1_auroc[i], head1_auroc[i], mid1_auroc[i], tail1_auroc[i]])
        x2.append([all2_auroc[i], head2_auroc[i], mid2_auroc[i], tail2_auroc[i]])
    for i in range(len(datasets)):
        x1.append([all1_fpr[i], head1_fpr[i], mid1_fpr[i], tail1_fpr[i]])
        x2.append([all2_fpr[i], head2_fpr[i], mid2_fpr[i], tail2_fpr[i]])
    # for i in range(len(datasets)):
    #     x1.append([head1[i], mid1[i], tail1[i]])
    #     x2.append([head2[i], mid2[i], tail2[i]])
    # print(x1)
    # exit()
    def y_fmt(x, y):
        return int(x)

    #####size
    large_size = 30
    mid_size = 26
    small_size = 22

    mpl.rcParams['axes.unicode_minus'] = False
    bar_width = 0.3
    handles=[]
    labels = []
    x = np.arange(4)
    plt.figure()
    figs, axes = plt.subplots(nrows=1, ncols=4, figsize=(32, 7))
    plt.setp(axes, xticks=[bar_width / 2, 1 + bar_width / 2, 2 + bar_width / 2, 3+ bar_width / 2], xticklabels=y)
    for i, ax in enumerate(axes.flatten()):
        if i <2:
            ax.bar(x, x1[i], bar_width, align='center', color='c', label='GradNorm', alpha=0.5, hatch='-')
            ax.bar(x + bar_width, x2[i], bar_width, align='center', color='orange', label='RP+GradNorm(Ours)',
                   alpha=0.5, hatch='/')
            # plt.bar(x+2*bar_width, x3, bar_width, align='center', color='r', label='~+Ours/IB'.format(name), alpha=0.5, hatch='x')

            ax.set_xlabel('Data Type', fontsize=large_size, fontweight='bold')
            ax.set_ylabel('AUROC (%)', fontsize=large_size, fontweight='bold')
            ax.legend(loc='upper left', fontsize=small_size)
            bottom = int(min(min(x1[i]), min(x2[i]))) - 1
            up = int(max(max(x1[i]), max(x2[i]))) + 2
            ax.set_ylim((bottom, up))
            ax.set_title('{}'.format(datasets[i%2]), fontsize=large_size, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=large_size)
            ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
        else:
            ax.bar(x, x1[i], bar_width, align='center', color='c', label='GradNorm', alpha=0.5, hatch='-')
            ax.bar(x + bar_width, x2[i], bar_width, align='center', color='orange', label='RP+GradNorm(Ours)',
                   alpha=0.5, hatch='/')
            # plt.bar(x+2*bar_width, x3, bar_width, align='center', color='r', label='~+Ours/IB'.format(name), alpha=0.5, hatch='x')

            ax.set_xlabel('Data Type', fontsize=large_size, fontweight='bold')
            ax.set_ylabel('FPR95 (%)', fontsize=large_size, fontweight='bold')
            if i == 2:
                ax.legend(loc='upper left', fontsize=small_size)
                up = int(max(max(x1[i]), max(x2[i]))) + 8
            else:
                ax.legend(loc='upper left', fontsize=small_size)
                up = int(max(max(x1[i]), max(x2[i]))) + 4
            bottom = int(min(min(x1[i]), min(x2[i]))) - 1
            ax.set_ylim((bottom, up))
            ax.set_title('{}'.format(datasets[i%2]), fontsize=large_size, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=large_size)
            ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    # plt.legend()



    # plt.ylabel('FPR95 (%)')
    # plt.ylim((40,90))
    # plt.ylim((35,100))
    # plt.ylim((0,100))
    # plt.xticks(size=16)
    # plt.yticks(size=16)
    # plt.xticks(x+bar_width, y, size=16)

    plt.tight_layout()
    plt.show()
    plt.savefig('LT_fig.pdf')
    plt.close()

    # vis_single([msp1, msp2, msp3], 'MSP', 'AUROC')
    # vis_single([msp1, msp2, msp3], 'MSP', 'FPR95')
    # vis_single([odin1, odin2, odin3], 'ODIN',  'AUROC')
    # vis_single([odin1, odin2, odin3], 'ODIN', 'FPR95')
    # vis_single([energy1, energy2, energy3], 'Energy', 'AUROC')
    # vis_single([energy1, energy2, energy3], 'Energy', 'FPR95')
    # vis_single([gradnorm1, gradnorm2, gradnorm3], 'GradNorm', 'AUROC')
    # vis_single([gradnorm1, gradnorm2, gradnorm3], 'GradNorm', 'FPR95')
