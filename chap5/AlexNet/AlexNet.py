import d2lzh as d2l
import mxnet as mx
from mxnet import gluon, autograd, nd, init
from mxnet.gluon import loss as gloss, data as gdata, nn
import os, sys

# AlexNet 网络结构
# 一共8层,即下面代码共8行.分别是(conv0,pool0), (conv1,pool1), (conv2), (conv3), (conv4,pool2), (dense0), (dense1), (dense2)
# channels和dense是人为给的,完全可以修改. 但是后面kernel_size,padding,strides,pool_size直接关系到下层网络的大小
# 首先应该清楚的是kernel_size对应的是conv,而pool_size对应的是pool
# 不难看出卷积共5层,第一个是11,第一个是5,第3,4,5都是３
# 有3层用了最大池化,参数完全一样(pool_size=3, strides=2),目的使得卷积出的尺寸和接下来的池化尺寸一样!!!
# 两个全连接层均为4096,且以0.5的概率丢弃
net = nn.Sequential()
net.add(nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'), nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'), nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'), nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(10))

# Test
X = nd.random.uniform(shape=(1,1,224,224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


# 读取数据集~/.mxnet/datasets/fashion-mnist/,之前已经下载了
# 将原来的28*28resize到224*224,主要是ImageNet训练耗时
def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join('~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root)         # 展开用户路径'~'
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]     # 将数据变换大小
    transformer += [gdata.vision.transforms.ToTensor()]             # 将uint8->float32,并且归一化到0-1
    transformer = gdata.vision.transforms.Compose(transformer)      # 串联两个变换

    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)  # 训练
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)  # 测试
    num_worker = 0 if sys.platform.startswith('win32') else 4       # 读取线程数目4
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=num_worker)
    test_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=False, num_workers=num_worker)
    return train_iter, test_iter

# 读取数据集
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

# 模型重初始化
ctx = d2l.try_gpu()
net.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)

# 优化函数
lr = 0.01
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})

# 训练模型
num_epochs = 15
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)













