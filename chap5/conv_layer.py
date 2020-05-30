from mxnet import  autograd, nd
from mxnet.gluon import nn

# 二位相关运算
def corr2d(X, K):
    h, w = K.shape
    output = nd.zeros((X.shape[0]-h+1, X.shape[1]-w+1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = (X[i:(i+h), j:(j+w)]*K).sum()
    return output
# 测试二维卷积运算
X = nd.array([[0,1,2], [3,4,5], [6,7,8]])
K = nd.array([[0,1], [2,3]])
print(corr2d(X,K))

# Demo-图像边缘检测
X = nd.ones((6, 8))
X[:, 2:6] = 0       #; print(X)
K = nd.array([[1, -1]])
Y = corr2d(X, K)
print('Demo1 图像垂直检测', Y)

# Demo-图像边缘检测
X1 = nd.ones((8, 6))
X1[2:6, :] = 0       #; print(X)
K1 = nd.array([[1, -1]]).reshape(-1, 1)     # 模板转成列向量
Y1 = corr2d(X1, K1)
print('Demo2 图像水平检测', Y1)

# 二维卷积层　这个类没有使用
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)                                  # 初始化父类
        self.weight = self.params.get('weight', shape=kernel_size)  # Cov2D成员变量weight
        self.bias = self.params.get('bias', shape=(1,))             # Cov2D成员变量bias
        self.kernel_size = kernel_size

    """ 前向计算　net(x)自动调用这个函数　"""
    def forward(self, X):
        # return nd.Convolution(X, self.weight.data(), self.bias.data(), self.kernel_size)  # need　NDArray type,but self.weight is conv2d0_weight (shape=(1, 2), dtype=<class 'numpy.float32'>)
        return corr2d(X, self.weight.data()) + self.bias.data()


# Demo-已知X,Y学习出K?
# 输出通道是1 核数组是(1, 2)
conv2d = nn.Conv2D(1, kernel_size=(1, 2))       # 直接使用nn的Conv2D
# conv2d = Conv2D(kernel_size=(1, 2))       # 直接使用nn的Conv2D
conv2d.initialize()

# 二位卷积使用4D数据(样本 通道 高 宽), 这里样本数为１,通道为１
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        l = (conv2d(X)-Y)**2        # conv2d(X)会自动调用父类的__call__　把数据丢进网络　计算结果和标准的平方损失
    l.backward()
    conv2d.weight.data()[:] -= 0.03*conv2d.weight.grad()    # SGD优化算法　这里只考虑权重的影响
    if (i+1)%2 == 0:
        print('batch %d, loss %f' % (i+1, l.sum().asscalar()))

print(conv2d.weight.data().reshape(1, -1))



