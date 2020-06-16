import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn

# ResNet网路结构, 一个节点: 两层3*3卷积+use_1x1conv(X)
# 和GoogLeNet相比,每个输出增加了批量归一化层
# 这里有两个问题, 如何保证HW不变;如何使XY通道数一致
# 如何保证HW不变: 这里use_1x1conv有些多余,因为strides不为１,肯定形状缩小了,必然使用1x1conv,直接用strides?1来判断
# 如何使XY通道数一致: 最好用X.shape[1]
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides) # 输入减到1/strides
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)                  # 不改变H W
        # 使用1*1卷积层:和两个卷积层输出形状配皮,同事方便下面判断
        if use_1x1conv:
            # 这里最好用X.shape[1]
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)        # 匹配卷积层,同样减到1/strides
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()                   # 归一化
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))        # 卷积->归一化->激活
        Y = self.bn2(self.conv2(Y))                 # 先卷积->归一化,把输入的加上再激活
        if self.conv3:
            X = self.conv3(X)                       # 有必要操作X
        return nd.relu(Y + X)


# Test
blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
X = nd.random.uniform(shape=(4, 3, 6, 6))
print(blk(X).shape)

# ResNet也采用了重复块的思想,这里的块节点是resnet_block,模型是18-layer,细节可以参考论文
# 由于3*3的池化层的strides=2,输出尺寸减半,所以下面有个判断(第一次就不要减半了),比如第一天上班有人给你搽了桌子,后面就自己搽了
# 通道数依次增加两倍,64->128->256->512
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:      #　处理第一次特殊情况
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk

# ResNet, 18-layer
net = nn.Sequential()
# 1+17=18层
net.add(nn.Conv2D(64, kernel_size=7, padding=3, strides=2), nn.BatchNorm(), nn.Activation('relu'), nn.MaxPool2D(pool_size=3, padding=1, strides=2))
# (2+2+2+2)*2+1=17层
# 18-layer 2 2 2 2
# 34-layer 3 4 6 3 效果不是太好
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2),
        nn.GlobalAvgPool2D(),       # 全局平均池化和全连接层
        nn.Dense(10))

# Test
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

# 重新初始化模型
ctx = d2l.try_gpu()
net.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)              # 模型重新初始化

# 优化函数, 0.1
lr = 0.1
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})  # 这行代码要放在模型初始化完成后,否则collect_params出错

# 训练
num_epochs = 5
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


