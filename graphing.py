from pac.data_gen import DataGen
import numpy as np
import matplotlib.pyplot as plt
from pac.data_gen2 import DataGen2
import seaborn as sns


def plot_heatmap(data, model, max_it):
    test_input = data.gen_input()  # 网格化输入数据
    test_label = data.gen_labels_all()  # 网格化的标签
    pre = model.evaluate(test_input)  # 训练好的network得到的预测值

    # --------生成热图---------##
    matrix_1 = (test_label - pre).reshape(data.zsteps, data.tsteps).detach().numpy()
    matrix_2 = np.log(matrix_1 ** 2)
    matrix_3 = matrix_1 ** 2

    cmap2 = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(matrix_3, cmap=cmap2, robust=True)
    plt.xlabel("z")
    plt.ylabel("t")
    plt.show()
    plt.savefig("D:\photosaved\loss_comprt_iterations%d_networksizeSmall.png" % max_it)
