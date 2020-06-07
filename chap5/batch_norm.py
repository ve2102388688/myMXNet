import d2lzh as d2l
from mxnet import autograd, init, gluon, nd
from mxnet.gluon import nn

# 批量归一化层, 共三块内容:全连接如何处理?卷积层如何处理?预测时如何处理?
# gamma, beta: 本层的参数
# moving_mean, moving_var:上一次的均值方差,预测单独处理
# eps, momentum: 防止归一化分母为0;移动平均-权衡本次和上一次
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not autograd.record():
        # 预测模式,直接使用传入的移动平均得到均值和方差
        X_hat = (X-moving_mean) / nd.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2, 4)       # 长度只能取2或４
        if len(X.shape) == 2:
            # 全连接层,计算特征维上(axis=0)的均值和方差
            mean = X.mean(axis=0)               # axis=0: 一axis=0axis=0列算一个均值
            var = ((X-mean) **2).mean(axis=0)
        else:
            # 二维卷基层,计算通道上(axis=1)的均值和方差
            mean = X.mean(axis=(0,2,3), keepdims=True)
            var = ((X-mean)**2).mean(axis=(0,2,3), keepdims=True)

        # 训练模式, 计算当前的均值和方差
        X_hat = (X-mean) / nd.sqrt(var+eps)

        # 移动平均-更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0-momentum)*mean
        moving_var = momentum * moving_var + (1.0-momentum)*var
    Y = gamma * X_hat + beta            # 拉伸和偏移
    return Y, moving_mean, moving_var


# num_dims=2/4,分别代表全链接和卷基层
# num_fearures,全连接对应输出个数;　卷积层对应输出通道数目
class BatchNorm(nn.Block):
    def __init__(self, num_fearures, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_fearures)           # 全连接层
        else:
            shape = (1, num_fearures, 1, 1)     # 二维卷基层
        # 参与求梯度和迭代的　拉伸和偏移,分别初始化为0或１
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', grad_req='null', shape=shape, init=init.Zero())
        # 不参与求梯度和迭代的变量,在内存上全吃书啊为0,先在内存上初始化之后搬到显存上
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)


    def forward(self, X):
        # 将moving_mean　moving_var复制到显存
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
        # 保存更新后的moving_mean,moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# LeNet,添加自己写的BatchNorm
# 全连接层和卷积层的归一化稍微有些不同, num_fearures,全连接对应输出个数;　卷积层对应输出通道数目(第一个参数跟着前面),第二个参数全连接2,卷积层4
# 批量归一化和Dropout一样,训练和测试不一样
# 批量归一化在激活之前,卷积/全连接之后
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4), nn.MaxPool2D(pool_size=2, strides=2), nn.Activation('sigmoid'),
        nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4), nn.MaxPool2D(pool_size=2, strides=2), nn.Activation('sigmoid'),
        nn.Dense(120), BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
        nn.Dense(60), BatchNorm(60, num_dims=2), nn.Activation('sigmoid'),
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


# 查看相关参数,gamma与beta
# net层编号从０开始,并且一个逗号加一层
# print('net[1] gamma and beta:\n', net[1].gamma.data().reshape((-1,)), net[1].beta.data().reshape((-1,)))


