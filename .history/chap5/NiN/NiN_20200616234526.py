import d2lzh as d2l
from mxnet import init, nd, gluon
from mxnet.gluon import data as gdata, nn 

# TODO NiN快:一个卷积层+两个1*1的卷积层
# FIXME 卷基层的超参数可以自行设置,比如,下面和AlexNet一致
# 两个1*1的卷基层代替全连接层,其参数一般是固定的
# 默认情况下padding=0, strides=1是显然的啊
def NiN_block(channels, kernel_size, padding=0, strides=1):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(channels, kernel_size, padding=padding, strides=strides, activation='relu'),
            # 两个1*1的卷基层代替全连接层,其参数一般是固定的
            # nn.Conv2D(channels, kernel_size=1, activation='relu'),
            nn.Conv2D(channels, kernel_size=1, activation='relu'))
    return blk


# TODO 定义NiN网络结构,模型大体和AlexNet一致
# 有3层用了最大池化,参数完全一样(pool_size=3, strides=2),目的使得卷积出 的尺寸和接下来的池化尺寸一样!!!
# NiN去除了容易造成过拟合的全连接输出层,替换成输出通道数等于标签类别数的NiN块和全局平均池化层,这种方式泛化能力相对低些,但是需要的显存大大降低
# 输出尺寸是输入的一半-(p,s)=(3,2),(pool_size,strides)
net = nn.Sequential()
net.add(NiN_block(channels=96, kernel_size=11, strides=4), nn.MaxPool2D(pool_size=3, strides=2),
        NiN_block(channels=256, kernel_size=5, padding=2), nn.MaxPool2D(pool_size=3, strides=2),
        NiN_block(channels=384, kernel_size=3, padding=1), nn.MaxPool2D(pool_size=3, strides=2), nn.Dropout(0.4),
        # 类别数10,下面的全局平均池化自动接任10
        NiN_block(channels=10, kernel_size=3, padding=1, strides=1),
        # 全局平均池化-poolsize就是输入大小,即输出(_*_*1*1)
        nn.GlobalAvgPool2D(),
        # 将输出4D->2D(N,10)
        nn.Flatten())

# Test
X = nd.uniform(shape=(1, 1, 224, 224))
net.initialize()       # 默认初始化init=initializer.Uniform()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)



# 读取数据
batch_size = 32
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

# 重新初始化模型
ctx = d2l.try_gpu()
net.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)              # 模型重新初始化

# 优化函数, 0.1
lr = 0.05
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})  # 这行代码要放在模型初始化完成后,否则collect_params出错

# 训练
num_epochs = 5
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


