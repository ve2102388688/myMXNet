import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn

# 卷积结构: 改良版的"批量归一化,激活,卷积",在ResNet中是"卷积,批量归一化,激活"
# 论文中还有1*1卷积层,这里忽略了
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),              # 批量归一化,激活
            nn.Conv2D(num_channels, kernel_size=3, padding=1))  # 卷积
    return blk

# 稠密快 稠密块由多个conv_block组成
# 前向计算时,每块的输入和输出在通道维上连结
# num_channels在下面的代码称为grouth_rate,形象的理解成每次增加的比例,近似成正比
# 问题:每模块增加num_convs*num_channels,极大增加了模型的复杂度
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X, Y, dim=1)      # 在通道维上连接,也就是说X.shape[1]+num_convs*num_channels,增加是很可怕的
        return X

# Test
blk = DenseBlock(2, 10)
blk.initialize()
X = nd.random.uniform(shape=(4, 3, 8, 8))
Y = blk(X)
print(Y.shape)


# 过渡层:防止模型过于复杂(上面存在的问题),两个手段:1*1卷积层,strides=2
#　1*1卷积层: 减少输出通道
#　strides=2: 平均池化成倍缩小输出尺寸
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),     # method1
            nn.AvgPool2D(pool_size=2, strides=2))       # method2
    return blk

# Test 输出通道立马从23变到10,你想怎么边怎么变,只有1*1卷积层能办到
blk = transition_block(10);
blk.initialize()
Y = blk(X)
print(Y.shape)


# DenseNet模型
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, padding=3, strides=2),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, padding=1, strides=2))

num_channels, grouth_rate = 64, 32                      # num_channels当前通道数,也就是上面的64
num_convs_in_dense_block = [4, 4, 4, 4]                 # 和论文不一样,如DenseNet-121(6,12,24,16),这里考虑到计算性能,全部取为4
for i, num_convs in enumerate(num_convs_in_dense_block):
    net.add(DenseBlock(num_convs, grouth_rate))
    num_channels += num_convs*grouth_rate               # 增加的通道数,这里每次增加4*32=128
    if i != len(num_convs_in_dense_block)-1:            # 相邻的DenseBlock有一个transition_block,即只有三个transition_block
        num_channels //= 2                              # 整除2,缩小输出尺寸
        net.add(transition_block(num_channels))

net.add(nn.BatchNorm(), nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))


# Test
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)



# 读取数据
batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

# 重新初始化模型
ctx = d2l.try_gpu()
net.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)              # 模型重新初始化

# 优化函数, 0.1
lr = 0.5
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})  # 这行代码要放在模型初始化完成后,否则collect_params出错

# 训练
num_epochs = 5
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)











