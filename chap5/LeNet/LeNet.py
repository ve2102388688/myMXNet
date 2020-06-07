import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, nd, init, gluon
from mxnet.gluon import loss as gloss, nn
import time

# 构建网络 LeNet:所有的层kernel_size=(5,5),且第一层6个通道,且第二层16个通道
# pooling: 两个池化层参数一致, 都是pool_size=2, strides=2
# 全连接层: 有三层,隐层单元分别是120,84,10
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'), nn.MaxPool2D(pool_size=2, strides=2),
       nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'), nn.MaxPool2D(pool_size=2, strides=2),
       # Dense默认将(N*C*H*W)->(N,C*H*W)
       # nn.Dense(120, activation='sigmoid'),
       nn.Dense(60, activation='sigmoid'),
       nn.Dense(10))

# 一个样本测试(1*1*32*32)
X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

# 尝试使用GPU
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()                  # 失败使用CPU
    return ctx

# Test try_gpu
ctx = try_gpu()
print(ctx)


def evaluate_accuray(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X,y in data_iter:
        # 如果有gpu,将数据复制到GPU
        X,y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')      # 将内存上的数据复制到显存
        acc_sum += (net(X).argmax(axis=1) == y).sum()                           # X表示有多个
        n += y.size
    return acc_sum.asscalar()/n

def train_ch5(train_iter, test_iter, net, loss, batch_size, num_epochs, trainer, ctx):
    print('Training on', ctx)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, startTime = 0.0, 0.0, 0, time.time()
        for X,y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)      # 将内存上的数据复制到显存
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)                                # 前进一部

            # 统计训练的平均误差
            train_l_sum += l.asscalar()
            y = y.astype('float32')
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuray(test_iter, net, ctx)
        print('epoch %d, loss %0.6f, train acc %0.4f, test_acc %0.4f, time %0.3f'
              % (epoch+1, train_l_sum/n, train_acc_sum/n, test_acc, time.time()-startTime))


# load data-fashion_mnist
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# loss function
loss = gloss.SoftmaxCrossEntropyLoss()

# reinit model
num_epochs, lr = 20, 2.5
net.initialize(force_reinit=True, ctx=ctx, init=init.Normal(sigma=0.2))              # 非第一次初始化要用force_reinit=True

# optimation
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})

# training
train_ch5(train_iter, test_iter, net, loss, batch_size, num_epochs, trainer, ctx)


