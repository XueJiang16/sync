import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_auroc = '''66.44
                    64.42
                    62.12
                    68.46
                    69.75
                    66.83
                    66.31
                    64.62
                    72.08
                    73.59
                    66.44
                    65.59
                    63.55
                    70.81
                    72.93
                    65.89
                    66.26
                    64.68
                    71.91
                    74.53
                    64.65
                    65.13
                    63.63
                    70.60
                    73.21
                    61.49
                    61.73
                    60.47
                    69.74
                    73.91
                    60.03
                    60.69
                    59.93
                    71.26
                    76.05
                    '''
    data_fpr = '''87.95
                    88.21
                    89.65
                    77.84
                    76.72
                    88.37
                    88.28
                    89.72
                    74.01
                    72.07
                    89.16
                    89.63
                    91.44
                    77.33
                    74.32
                    90.36
                    90.63
                    91.26
                    77.11
                    72.42
                    90.50
                    91.36
                    92.20
                    78.92
                    74.92
                    94.23
                    94.43
                    94.85
                    81.57
                    74.50
                    95.44
                    95.52
                    95.30
                    80.45
                    70.12
                    '''



    data_list = list(map(float, data_auroc.split()))
    # data_list = list(map(float, data_fpr.split()))
    msp = data_list[::5]
    odin = data_list[1::5]
    energy = data_list[2::5]
    gradnorm = data_list[3::5]
    ours = data_list[4::5]
    from matplotlib.ticker import MaxNLocator



    y = ['{}'.format(i) for i in range(2,9)]
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('lightsteelblue')
    ax.patch.set_alpha(0.1)

    plt.plot(y, msp, label='MSP', marker="d")
    plt.plot(y, odin, label='ODIN', marker="x")
    plt.plot(y, energy, label='Energy', marker="p")
    plt.plot(y, gradnorm, label='GradNorm', marker="v")
    plt.plot(y, ours, label='RP+GradNorm(Ours)', marker="*")
    plt.legend(loc='upper left', fontsize=14, ncol=2, framealpha=0.5)
    plt.xlabel('Tail Index a', fontsize=18, fontweight='bold')
    plt.ylabel('Average AUROC (%)', fontsize=18, fontweight='bold')
    plt.ylim((59, 81))
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.grid(color='w')
    ax = plt.gca()  # 获取当前的axes
    ax.spines['right'].set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['bottom'].set_color('w')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()
    plt.savefig('line_auroc.pdf')

