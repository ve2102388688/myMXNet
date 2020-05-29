import d2lzh as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss

# 定义丢弃方法
def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob       # 保留的概率
    # 此种情况全部删除
    if keep_prob==0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob     # 按照X的形状生成一个矩阵 且小于keep_prob为1,否则为0
    return mask*X/keep_prob                     # 保留的数据以1/(1-p)概率拉伸
# 测试丢弃0 0.5 1的效果
X = nd.arange(16).reshape((2, 8))
print(dropout(X,0))                 # 0的概率丢弃
print(dropout(X,0.5))               # 0.5的概率丢弃
print(dropout(X,1))                 # 1的概率丢弃


# 定义模型参数-两层感知机
num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hidden1))
b1 = nd.zeros(num_hidden1)
W2 = nd.random.normal(scale=0.01, shape=(num_hidden1, num_hidden2))
b2 = nd.zeros(num_hidden2)
W3 = nd.random.normal(scale=0.01, shape=(num_hidden2, num_outputs))
b3 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2, W3, b3]
for param in params:                # 分配梯度内存空间
    param.attach_grad()

# 定义网络
drop_prob1, drop_prob2 = 0.2, 0.5
def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training():              # 只在训练的时候丢弃
        H1 = dropout(H1, drop_prob1)        # 在第一次全连接层丢弃
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():              # 只在训练的时候丢弃
        H2 = dropout(H2, drop_prob2)        # 在第而次全连接层丢弃
    return nd.dot(H2, W3)+b3

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

# 训练
num_epochs, lr = 100, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr, None)

