import torch
from pac.net2 import *
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt
from pac.data_gen2 import DataGen2
import seaborn as sns

if __name__ == "__main__":
    # 输出数据的初始化
    zs = 0.1
    ze = 10
    ts = 1
    te = 3000
    zsteps = 10
    tsteps = 300
    batch_size = 100
    # 生成数据

    data = DataGen2(zs, ze, ts, te, zsteps=zsteps, tsteps=tsteps)
    dim_in = 2
    dim_out = 1
    # hidden_list = (50, 40, 30, 30, 20)
    hidden_list = (20, 10, 8, 8, 5)
    model_name = 'DNN'
    init_lr = 0.01
    max_it = 1000

    # 神经网络各组分的初始化
    # model = DNN1(indim=dim_in, outdim=dim_out, hidden_units=hidden_list, name2Model=model_name, actName2in='tanh',
    #                 actName='tanh')
    # model = Pure_DenseNet(indim=dim_in, outdim=dim_out, hidden_units=hidden_list, name2Model=model_name, actName2in='tanh',
    #                 actName='tanh')
    model = PDE_DNN(dim_in=dim_in, dim_out=dim_out, hidden_layers=hidden_list, name2Model=model_name, actName_in='tanh',
                    actName_hidden='tanh')
    params2Net = model.DNN1.parameters()
    optimizer = torch.optim.Adam(params2Net, lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.995)
    arr2epoch = []
    arr2loss = []
    arr2lr = []
    predi = []
    loss_append = []
    for i_epoch in range(max_it):

        # 每次迭代都随机生成数据
        it_inputs, it_labels = data.gen_inter_dz(batch_size)
        bd_inputs, bd_labels = data.gen_bound(batch_size)
        loss = model.loss_it_net(it_inputs, it_labels) + model.loss_it_pde_dt(it_inputs, it_labels) \
               + model.loss_bd_net(bd_inputs, bd_labels)
        loss_append.append(loss.item())
        prediction = model.evaluate(it_inputs)
        predi.append(prediction)
        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()  # 求偏导
        optimizer.step()  # 更新参数
        scheduler.step()

        if i_epoch % 100 == 0:
            print('i_epoch --- loss:', i_epoch, loss.item())
            # print("第%d个epoch的学习率：%f" % (i_epoch, optimizer.param_groups[0]['lr']))
            arr2loss.append(loss.item())
            arr2lr.append(optimizer.param_groups[0]['lr'])

    # -------------loss----------#
    logloss = np.log(loss_append)
    plt.figure()
    plt.title("log(loss) trend")
    plt.plot(logloss, color='b', label='Label')
    plt.xlabel("iterations")
    plt.ylabel("log(loss)")
    plt.show()

    # # ---gen_data----#
    test_input = data.gen_input()  # 网格化输入数据
    test_label = data.gen_labels_all_dz()  # 网格化的标签
    pre = model.evaluate(test_input)  # 训练好的network得到的预测值

    # --------生成热图---------##
    matrix_1 = (test_label - pre).reshape(zsteps, tsteps).detach().numpy()
    matrix_2 = np.log(matrix_1 ** 2)
    matrix_3 = matrix_1 ** 2

    plt.figure()
    cmap2 = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    heatmap = sns.heatmap(matrix_3, cmap=cmap2, robust=True)
    plt.show()
