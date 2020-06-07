import d2lzh as d2l
from mxnet import init, gluon
from mxnet.gluon import nn

# 定义LeNet模型,加入批量归一化
# 批量归一化在激活之前,卷积/全连接之后
# 批量归一化和Dropout一样,训练和测试不一样
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5), nn.BatchNorm(scale=False), nn.Activation('sigmoid'), nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5), nn.BatchNorm(scale=False), nn.Activation('sigmoid'), nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
        nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
        nn.Dense(10))

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型
ctx = d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

# 优化算法
lr = 2
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})

# 训练模型
num_epochs = 5
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


