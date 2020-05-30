import d2lzh as d2l
from mxnet import autograd, nd
import time

start = time.time()
batch_size = 256                      # 一次处理快大小
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 初始化模型参数
num_inputs = 28*28
num_outputs = 10
# W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
W = nd.zeros(shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
W.attach_grad()            # 提前分配内存
b.attach_grad()            # 提前分配内存


# softmax运算
# # 一个测试案例-按列/列求和
# X = nd.array([[1,2,3],[4,5,6]])
# print(X.sum(axis=0, keepdims=True))                # 按列求和
# print(X.sum(axis=1, keepdims=True))                # 按行求和

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)       # 按行求和,结果列向量
    return X_exp/partition

# # 一个测试案例-调用softmax
# X = nd.random.normal(scale=0.01, shape=(3,5))
# print(X)
# print(softmax(X))                                      # 讲数据归一化到概率形式 即每个值都大于0且和为1
# print(softmax(X).sum(axis=1))                           # 验证行和为1


# 定义模型
# 将4维空间转换成2维 N*W*H*D -1系统自动计算
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)

# 定义交叉熵损失
# # 一个测试案例-pick()根据下标取元素的值,注意第二参数是下标 第一个参数是原始数据
# y_hat = nd.array([[0.1,0.3,0.6],[0.3,0.2,0.5]])
# y = nd.array([0,2], dtype='int32')     # print(y, '\t', y.dtype)
# print(nd.pick(y_hat, y))               # output [0.1 0.5]

# 从上面可以看出 y是下标,因此和上类似,只是在输出都套了一层log,注意 每次只有一个1
def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()
# print(cross_entropy(y_hat, y))


# 定义模型 比较后求平均 该函数没有使用 可以忽略
def accurary(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
# print(accurary(y_hat, y))

# 分类准确率 y_hat和y相等为1,不等为0
def evaluate_accurary(data_iter, net):
    acc_sum, n = 0.0, 0
    for X,y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size                            # 一批量是256
    return acc_sum/n
print('初始模型准确率: ', evaluate_accurary(test_iter, net))       # 测试集 准确率大概0.1(1/10)因为W,b是随机的e


# 训练模型 <数据 网络 迭代次数 批量大小 跟新参数 学习率 学习器>
num_epochs, lr = 5, 0.05
def train_softmax(train_iter, test_iter, net, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0    # 损失函数（逐渐减小的） 训练集正确率 样本数目60000
        for X,y in train_iter:
            with autograd.record():
                y_hat = net(X)             # 训练模型 输出<NDArray 256x10 @cpu(0)>
                l = loss(y_hat, y).sum()                                   # 结果是256的行向量 需要求和,也可以放在l.asscalar(),等效的
            l.backward()                                                   # 如果上面没求和,这里会自动求和的
            if trainer is None:
                d2l.sgd(params, lr, batch_size)                            # 第一次用sgd设置参数
            else:
                trainer.step(batch_size)                                   # 其他前进一步训练
            train_l_sum += l.asscalar()
            # 训练集正确率（累计）
            y = y.astype('float32')
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size                                                    # 一次加上256,总和60000
        test_acc = evaluate_accurary(test_iter, net)                       # 测试集正确率
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc)) # 交叉熵损失 训练集正确率 测试集正确率

train_softmax(train_iter, test_iter, net, cross_entropy, num_epochs, batch_size, [W,b], lr)


# 预测 就一个批量数据块即可
for X,y in test_iter:
    break
true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())                        # 获取真实标签
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())    # 获取预测标签
title = [true + '\n' + pred for true,pred in zip(true_labels, pred_labels)]    # 融合标签
d2l.show_fashion_mnist(X[0:9], title[0:9])
# d2l.plt.show()                                 # 不加上不会显示图片

print('total time:', time.time()-start)

