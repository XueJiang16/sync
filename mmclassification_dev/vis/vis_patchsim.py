import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





if __name__ == '__main__':
    patchsim_id = np.load('./patchsim_imagenet.npy')
    patchsim_ood = np.load('./patchsim_Textures.npy')
    sns.set(rc={'figure.figsize': (8, 6)})
    # sns.set_style('whitegrid')
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    fig = sns.kdeplot(np.array(patchsim_id), bw=0.2)
    sns.kdeplot(np.array(patchsim_ood), bw=0.2)
    plt.legend(labels=['ID', 'OOD'])
    plt.xlabel('Patch Sim', fontsize=18, fontweight='bold')
    plt.ylabel('Density', fontsize=18, fontweight='bold')
    plt.yticks(size=16)
    plt.xticks(size=16)
    plt.tight_layout()
    fig.get_figure().savefig('test.jpg')
    plt.close()
