from mxnet import nd
from mxnet.gluon import nn


"""**************************** padding使得输入输出同型 ****************************************"""
# 计算卷积层　先初始化参数
def comp_conv2d(con2d, X):
    con2d.initialize()
    X = X.reshape((1,1) + X.shape)      # (1,1)表示批量大小和通道
    Y = con2d(X)
    return Y.reshape(Y.shape[2:])       # 剔除前面两维:批量大小和通道

# Demo1 padding=1((kernel_size-1)/2),意思是一共添加2行,可保证输入输出一样(奇数)
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)     # channels=1(first paramener)
X = nd.random.uniform(shape=(8,8))
print('Demo1:', comp_conv2d(conv2d, X).shape)         # 观察输出的形状　和输入一致

# Demo2 改变padding使得输入输出同型
conv2d = nn.Conv2D(1, kernel_size=(5,3), padding=(2,1))     # channels=1(first paramener)
print('Demo2:', comp_conv2d(conv2d, X).shape)         # 观察输出的形状　和输入一致


"""**************************** stride使得输出为输入的1/n ****************************************"""
# Demo3 改变padding使得输入输出同型 进一步使得输出减半strides=2
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)     # channels=1(first paramener)
print('Demo3:', comp_conv2d(conv2d, X).shape)         # 观察输出的形状　和输入一致

# Demo4 (X.shape-kernel_size+2*padding+strides)/strides
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))     # channels=1(first paramener)
print('Demo4:', comp_conv2d(conv2d, X).shape)         # 观察输出的形状　和输入一致

