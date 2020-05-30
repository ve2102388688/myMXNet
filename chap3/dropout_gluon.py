import d2lzh as d2l
from mxnet import autograd, gluon, init
from mxnet.gluon import loss as gloss, nn

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义网络病初始化
drop_prob1, drop_prob2 = 0.5, 0.2
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dropout(drop_prob1),   # 在第一次全连接层丢弃
        nn.Dense(256, activation='relu'), nn.Dropout(drop_prob1),   # 在第一次全连接层丢弃
        nn.Dense(256, activation='relu'), nn.Dropout(drop_prob1),   # 在第一次全连接层丢弃
        nn.Dense(256, activation='relu'), nn.Dropout(drop_prob2), nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

#################################### 丢弃法  ####################################################
# 优化算法
lr = 0.5
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':lr})

# 训练
num_epochs = 15
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)



#################################### 权重丢弃法  ####################################################
# # 定义模型
# net = nn.Sequential()
# net.add(nn.Dense(256, activation='relu'), nn.Dense(256, activation='relu'), nn.Dense(10))     # 'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'
# net.initialize(init.Normal(sigma=0.01))
#
# # 分类准确率 y_hat和y相等为1,不等为0
# def evaluate_accurary(data_iter, net):
#     acc_sum, n = 0.0, 0
#     for X,y in data_iter:
#         y = y.astype('float32')
#         acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
#         n += y.size                            # 一批量是256
#     return acc_sum/n
# print('初始模型准确率: ', evaluate_accurary(test_iter, net))       # 测试集 准确率大概0.1(1/10)因为W,b是随机的e
#
# # 进对权重衰减 权重名字一般以weight结尾
# wd = 0
# trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd', {'learning_rate': lr, 'wd': wd})
# # 不对偏差衰减 权重名字一般以bias结尾
# trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd', {'learning_rate': lr})
#
# # 定义L2惩罚
# def l2_penalty(w):
#     return (w**2).sum()/2                      # 只惩罚模型权重参数
#
# for epoch in range(num_epochs):
#     train_l_sum, train_acc_sum, n = 0.0, 0.0, 0    # 损失函数（逐渐减小的） 训练集正确率 样本数目60000
#     for X,y in train_iter:
#         with autograd.record():
#             y_hat = net(X)             # 训练模型 输出<NDArray 256x10 @cpu(0)>
#             l = loss(y_hat, y).sum()                                   # 结果是256的行向量 需要求和,也可以放在l.asscalar(),等效的
#         l.backward()                                                   # 如果上面没求和,这里会自动求和的
#         trainer_w.step(batch_size)
#         trainer_b.step(batch_size)                                   # 其他前进一步训练
#         train_l_sum += l.asscalar()
#         # 训练集正确率（累计）
#         y = y.astype('float32')
#         train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
#         n += y.size                                                    # 一次加上256,总和60000
#     test_acc = evaluate_accurary(test_iter, net)                       # 测试集正确率
#     print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
#           % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc)) # 交叉熵损失 训练集正确率 测试集正确率


