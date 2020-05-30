from mxnet import autograd, nd
from mxnet import init                      # 初始化模型参数
from mxnet import gluon                     # 优化算法
from mxnet.gluon import data as gdata       # data包读取数据
from mxnet.gluon import nn                  # 神经网络模块
from mxnet.gluon import loss as gloss       # 损失函数

# 生成数据集
num_inputs = 2;                                                         # 特征数
num_samples = 1000;                                                     # 样本数
true_w = [2, -3.4];
true_b = 4.2;
features = nd.random.normal(scale=1, shape=(num_samples, num_inputs));  # 1000*2
labels = nd.dot(features, nd.transpose(nd.array(true_w))) + true_b;     # y = X*W + b
labels += nd.random.normal(0, 0.01, shape=labels.shape);                # y = X*W + b + noise

# 小批量读取数据集
batch_size = 10;
dataset = gdata.ArrayDataset(features, labels);                         # 将训练数据的features, labels组合在一起
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True);        # 随机读取批量小数据 随机打乱数据

# for X,y in data_iter:
#     print(X, y);
#     break;

# 定义模型
net = nn.Sequential();                          # 串联各个层的容器
net.add(nn.Dense(1));                           # 线性回归是单层的全连接层 Dense表示全连接 1层

# 初始化模型参数
net.initialize(init.Normal(sigma=0.01));        # 模型参数初始化

# 定义损失函数
loss = gloss.L2Loss();                          # 2范数定义损失函数

# 定义优化函数-net.collect_params() 网络层中所有参数
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.03});

# train samples
num_epochs = 5;     # 跑多少次training data
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y).mean();
        l.backward();                           # 相当于 l.sun().backward();
        trainer.step(1);               # 迭代模型参数 batch_size是求样本梯度平均
    l = loss(net(features), labels);            # 用训练的模型net去预测
    print('epoch=%d, train_loss=%f' % (epoch + 1, l.mean().asnumpy()));

print('\nresult:');
print(true_w, net[0].weight.data());            # 真实w 训练的w
print(true_b, net[0].bias.data());              # 真实b 训练的b

