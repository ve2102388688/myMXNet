import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
import time

start = time.time()

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义模型
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(256, activation='relu'), nn.Dense(10))     # 'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

# 优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.5})

# 训啦模型
num_epochs = 100
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)


print('total time:', time.time()-start)

