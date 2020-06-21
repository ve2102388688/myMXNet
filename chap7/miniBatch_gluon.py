import d2lzh as d2l
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn, loss as gloss, data as gdata
import numpy as np
import time


# TODO 读取数据并归一化，将数据分成两部分，前5列+最后一列
def get_data_ch7():
    data = np.genfromtxt('/home/topeet/Desktop/d2l/d2l-zh/data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:1500, :-1]), nd.array(data[:1500, -1]) # 前面5列一组，最后一列一组
features, labels = get_data_ch7()


def train_gluon_ch7(trainer_name, hyperparams, features, labels, num_epochs=2, batch_size=10):
    # 初始化模型 
    # TODO y = XW + b
    net, loss = nn.Sequential(), gloss.L2Loss()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))

    # 损失函数，用features，labels学出W,b
    def eval_loss():
        return loss(net(features), labels).mean().asscalar()

    ls = [eval_loss()]      # 记录损失变化
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    # 用Trainer来迭代参数
    trainer = gluon.Trainer(net.collect_params(), trainer_name, hyperparams)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X,y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)                    # 在trainer做平均
            trainer.set_learning_rate(0.1)
            if (batch_i+1)*batch_size % 100 == 0:       # 每100次记录下
                ls.append(eval_loss())

    print('loss: %f, %f sec per epoch' % (ls[-1], time.time()-start))
    d2l.set_figsize(figsize=(15,5))
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)   # loss曲线
    # 坐标轴
    d2l.plt.xlabel('epochs')
    d2l.plt.ylabel('loss')



train_gluon_ch7('sgd', {'learning_rate':0.05}, features, labels, batch_size=10)



d2l.plt.show()

