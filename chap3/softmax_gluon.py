import d2lzh as d2l
from mxnet import gluon,init
from mxnet.gluon import loss as gloss, nn

# 获取和读取数据
batch_size = 512
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义和初始化模型
net = nn.Sequential()          # 串联各个层的容器
net.add(nn.Dense(10))          # Dense全连接 输出个数10
net.initialize(init.Normal(sigma=0.01))

# softmax和交叉熵损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

# 定义优化算法
# 学习率为0.1的小批量随机梯度下降作为优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':2})

# 训练模型 和前面的一致
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)



# 预测 就一个批量数据块即可
for X,y in test_iter:
    break
true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())                        # 获取真实标签
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())    # 获取预测标签
title = [true + '\n' + pred for true,pred in zip(true_labels, pred_labels)]    # 融合标签
d2l.show_fashion_mnist(X[0:9], title[0:9])
# d2l.plt.show()                                 # 不加上不会显示图片


