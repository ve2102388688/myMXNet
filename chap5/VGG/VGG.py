import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn

# VGG重复网络单元
def vgg_block(num_covs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_covs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu'))   # 尺寸保存不变
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))                   # 尺寸减半
    return blk

# VGG模型
def vgg(conv_arch):
    net = nn.Sequential()
    # 卷积部分
    for (num_covs, num_channels) in conv_arch:
        net.add(vgg_block(num_covs, num_channels))
    # 全连接部分,两层4096,大约要占1G显存
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

# Test
conv_arch11 = ((1,64), (1,128), (2,256), (2,512), (2, 512))         # VGG11
conv_arch16 = ((2,64), (2,128), (3,256), (3,512), (3, 512))         # VGG16
conv_arch19 = ((2,64), (2,128), (4,256), (4,512), (4, 512))         # VGG19
net = vgg(conv_arch19)
net.initialize()        # 默认初始化init=initializer.Uniform()
X = nd.random.normal(shape=(1, 1, 96, 96))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)



# 读取数据集
batch_size = 32
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

# 模型重初始化
ctx = d2l.try_gpu()
net.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)              # 模型重新初始化

# 优化函数
lr = 0.05
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})  # 这行代码要放在模型初始化完成后,否则collect_params出错

# 训练模型
num_epochs = 5
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)



