import d2lzh as d2l
from mxnet import init, gluon, nd
from mxnet.gluon import nn

# Inception模块,共4条并行线路,Attention:这4条并行线路都没有改变尺寸大小,接着看看怎么实现这种技术
# 保证尺寸不变,首先肯定strides=1(Conv2D是默认值,代码中没有这项),因为strides会让输出以strides倍下降
# 1*1卷积层: 熙然不会改变尺寸大小,改变的输出通道数目C(前提是设置的通道数小于原始通道数)
# 3*3卷积层: (k,p)=(3, 1), (kernel_size, padding)
# 5*5卷积层: (k,p)=(5, 2), (kernel_size, padding)
# Inception块输出是通道的合并!!concat(xxx,dim=1)
class Inception(nn.Block):
    """ c1, c2, c3, c4 每条线路的输出通道数,总输出通道c1+c2+c3+c4,这些参数全是超惨 """
    def __init__(self,  c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        # 线路1:　1*1卷积层
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # 线路2: 1*1卷积层       ->3*3卷积层
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1, activation='relu')
        # 线路3: 1*1卷积层       ->5*5卷积层
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2, activation='relu')
        # 线路4: 3*3MaxPooling  ->1*1卷积层
        self.p4_1 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)     # 默认strides=None
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')
        # """ 加上BatchNorm """
        # # 线路1:　1*1卷积层
        # self.p1_1 = nn.Sequential().add(nn.Conv2D(c1, kernel_size=1), nn.BatchNorm(), nn.Activation('relu'))
        # # 线路2: 1*1卷积层       ->3*3卷积层
        # self.p2_1 = nn.Sequential().add(nn.Conv2D(c2[0], kernel_size=1), nn.BatchNorm(), nn.Activation('relu'))
        # self.p2_2 = nn.Sequential().add(nn.Conv2D(c2[1], kernel_size=3, padding=1), nn.BatchNorm(), nn.Activation('relu'))
        # # 线路3: 1*1卷积层       ->5*5卷积层
        # self.p3_1 = nn.Sequential().add(nn.Conv2D(c3[0], kernel_size=1), nn.BatchNorm(), nn.Activation('relu'))
        # self.p3_2 = nn.Sequential().add(nn.Conv2D(c3[1], kernel_size=5, padding=2), nn.BatchNorm(), nn.Activation('relu'))
        # # 线路4: 3*3MaxPooling  ->1*1卷积层
        # self.p4_1 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)     # 默认strides=None
        # self.p4_2 = nn.Sequential().add(nn.Conv2D(c4, kernel_size=1), nn.BatchNorm(), nn.Activation('relu'))

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return nd.concat(p1, p2, p3, p4, dim=1)     # 在通道维上连接输出,即dim=1是通道维度,在这个维度上合并


# GoogLeNet网络 共有5个模块
# 模块1 显然看见两个strides=2,因此输出是原来的1/4(96/4=24)
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, padding=3, strides=2, activation='relu'), nn.MaxPool2D(pool_size=3, padding=1, strides=2))
# 模块2 一个strides=2,因此输出是原来的1/２(24/2=12)
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),             # 1*1卷积层可以保持输出于输入尺寸一样
       nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'), nn.MaxPool2D(pool_size=3, padding=1, strides=2))
# 模块3 两个Inception块, (Inception中参数论文中有), output=480
b3 = nn.Sequential()
b3.add(Inception(64, (96,128), (16,32), 32),                        # output:64+128+32+32=256
       Inception(128, (128,192), (32,96), 64),                      # output:128+192+96+64=480
       nn.MaxPool2D(pool_size=3, padding=1, strides=2))
# 模块4, 5个Inception块, output=832
b4 = nn.Sequential()
b4.add(Inception(192, (96,208), (16,48), 64),                       # output:192+208+48+64=512
       Inception(160, (112,224), (24,64), 64),                      # output:160+224+64+64=512
       Inception(128, (128,256), (24,64), 64),                      # output:128+256+64+64=512
       Inception(112, (144,288), (32,64), 64),                      # output:112+288+64+64=528
       Inception(256, (160,320), (32,128), 128),                    # output:256+320+128+128=832
       nn.MaxPool2D(pool_size=3, padding=1, strides=2))
# 模块5, 两个Inception块, output=1024
b5 = nn.Sequential()
b5.add(Inception(256, (160,320), (32, 128), 128),                   # output:256+320+128+128=832
       Inception(384, (192,384), (48,128), 128),                    # output:384+384+128+128=1024
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))                           # 串联在一起


# Test
X = nd.random.uniform(shape=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)



# 读取数据
batch_size = 32
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


