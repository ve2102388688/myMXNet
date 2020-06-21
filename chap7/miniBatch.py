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

features, labels = get_data_ch7()       # (1500, 5)
print(features.shape)


# TODO 优化算法 这里不用除以批量大小,在with已经处理了
# hyperparams 超参
def sgd(params, states, hyperparams):
    for param in params:
        param[:] -= hyperparams['lr'] * param.grad


# 训练数据
def train_ch7(trainer_fn, states, hyperparams, features, labels, batch_size=10, num_epochs=2):
    # 初始化模型 
    # TODO y = XW + b
    net, loss = d2l.linreg, d2l.squared_loss
    W = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    W.attach_grad() # 提前分配内存
    b.attach_grad()

    # 损失函数，用features，labels学出W,b
    def eval_loss():
        return loss(net(features, W, b), labels).mean().asscalar()

    ls = [eval_loss()]      # 记录损失变化
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels), batch_size, shuffle=True)    # 读取训练数据集
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X,y) in enumerate(data_iter):     # batch_i只是一个计数变量
            with autograd.record():
                l = loss(net(X, W, b), y).mean()        # 平均损失
            l.backward()
            trainer_fn([W, b], states, hyperparams)     # 优化算法
            if (batch_i+1)*batch_size % 100 == 0:       # 每100次记录下
                ls.append(eval_loss())
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time()-start))

    d2l.set_figsize(figsize=(15,5))
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)   # loss曲线
    # 坐标轴
    d2l.plt.xlabel('epochs')
    d2l.plt.ylabel('loss')


def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(sgd, None, {'lr':lr}, features, labels, batch_size, num_epochs)


train_sgd(1, 1500, 6)   # batch_size=样本数　标准的梯度下降

train_sgd(0.005, 1)     # batch_size=1　随机梯度下降

train_sgd(0.05, 10)     # miniBatch,效果介于两者之间



d2l.plt.show()


