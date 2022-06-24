import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def vis_single(data_list, name, mode='AUROC'):
    y = ['iNaturalist', 'SUN', 'Places', 'Textures']
    if mode == 'FPR95':
        x1 = data_list[0][1::2]
        x2 = data_list[1][1::2]
        x3 = data_list[2][1::2]
    else:
        x1 = data_list[0][::2]
        x2 = data_list[1][::2]
        x3 = data_list[2][::2]

    mpl.rcParams['axes.unicode_minus'] = False
    bar_width = 0.2
    x = np.arange(4)
    plt.figure()
    plt.bar(x, x1, bar_width, align='center', color='c', label='~/B', alpha=0.5, hatch='-')
    plt.bar(x+bar_width, x2, bar_width, align='center', color='orange', label='~/IB'.format(name), alpha=0.5, hatch='/')
    plt.bar(x+2*bar_width, x3, bar_width, align='center', color='r', label='~+Ours/IB'.format(name), alpha=0.5, hatch='x')

    plt.xlabel('OOD datasets', fontsize=24, fontweight='bold')
    plt.ylabel('{} (%)'.format(mode), fontsize=24, fontweight='bold')
    # plt.ylabel('FPR95 (%)')
    # plt.ylim((40,90))
    # plt.ylim((35,100))
    bottom = int(min(min(x1), min(x2), min(x3))) - 1
    up = int(max(max(x1), max(x2), max(x3))) + 3
    plt.ylim((bottom, up))
    # plt.xticks(size=16)
    plt.yticks(size=20)
    plt.xticks(x+bar_width, y, size=20)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.16), ncol=3, fontsize=14)
    # plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.savefig('{}_{}_2.pdf'.format(name, mode))
    plt.close()

if __name__ == '__main__':
    data1 = '''83.70 	63.34 	81.62 	60.57 	75.59 	74.22 	58.39 	81.61 
    73.02 	88.24 	70.59 	83.44 	67.01 	88.18 	51.96 	94.93 
    72.41 	87.83 	75.81 	78.42 	71.04 	84.70 	46.20 	93.74 
    69.95 	88.62 	77.23 	76.65 	72.06 	83.89 	44.84 	92.96'''
    """gradnorm, msp, odin, energy """
    # imagenet-10%
    data1 = data1.split('\n')
    gradnorm1, msp1, odin1, energy1 = [list(map(float, x.split())) for x in data1]

    data2 = '''63.95 	97.72 	66.60 	93.13 	66.84 	92.11 	42.74 	98.79 
    64.95	96.44	67.39	91.79	67.46	91.16	43.05	98.51
    60.14 	98.70 	70.63 	93.13 	70.14 	91.96 	41.83 	98.30 
    86.66	93.85	71.59	97.67	67.56	97.24	68.04	95.37	 
    55.99 	98.74 	71.12 	93.11 	70.24 	91.30 	42.38 	98.07 
    91.92 	37.89 	80.81 	76.22 	77.15 	81.18 	64.48 	86.19 
    82.51 	72.19 	74.57 	78.10 	70.67 	86.58 	57.31 	84.95 
    91.23 	43.87 	77.36 	73.53 	72.67 	83.29 	62.94 	79.80'''

    # imagenet-a8
    data2 = data2.split('\n')
    msp2, msp3, odin2, odin3, energy2, energy3, gradnorm2, gradnorm3 \
        = [list(map(float, x.split())) for x in data2]
    #
    # vis_single([msp1, msp2, msp3], 'MSP', 'AUROC')
    # vis_single([msp1, msp2, msp3], 'MSP', 'FPR95')
    # vis_single([odin1, odin2, odin3], 'ODIN',  'AUROC')
    # vis_single([odin1, odin2, odin3], 'ODIN', 'FPR95')
    # vis_single([energy1, energy2, energy3], 'Energy', 'AUROC')
    vis_single([energy1, energy2, energy3], 'Energy', 'FPR95')
    # vis_single([gradnorm1, gradnorm2, gradnorm3], 'GradNorm', 'AUROC')
    vis_single([gradnorm1, gradnorm2, gradnorm3], 'GradNorm', 'FPR95')
