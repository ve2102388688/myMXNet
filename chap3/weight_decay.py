import d2lzh as d2l
from mxnet import autograd, nd, init,gluon
from mxnet.gluon import loss as gloss, data as gdata, nn

# 生成数据集或读取数据集
n_train, n_test, n_feature = 20, 100, 200
true_w, true_b = nd.ones((n_feature, 1))*0.01, 0.5

# 生成y
features = nd.random.normal(scale=1, shape=(n_train+n_test, n_feature))
labels = nd.dot(features, true_w) + true_b
labels += nd.normal(scale=0.01, shape=labels.shape)

# 生成batch data
batch_size = 1
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)

# 初始化模型参数
def init_params():
    w = nd.random.normal(scale=1, shape=(n_feature, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]

# 定义L2惩罚
def l2_penalty(w):
    return (w**2).sum()/2                      # 只惩罚模型权重参数

# 训练和测试
num_epochs, lr = 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss       # 适用线性网络 平方损失
def pit_and_plot(lambd):
    w,b = init_params()                        # 初始化模型参数
    train_l, test_l = [], []
    for _ in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                # 增加L2番薯惩罚
                l = loss(net(X, w, b), y) + lambd*l2_penalty(w)
            l.backward()
            d2l.sgd([w,b], lr, batch_size)
        train_l.append(loss(net(train_features, w, b), train_labels).mean().asscalar())
        test_l.append(loss(net(test_features, w, b), test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs+1), train_l, 'epochs', 'loss',
                 range(1, num_epochs+1), test_l, ['train', 'test'], figsize=(15, 5))
    print('L2 norm of w:', w.norm().asscalar())

pit_and_plot(lambd=10)      # 出现过拟合 测试集误差大
# pit_and_plot(lambd=3)      # 过拟合得到缓解 权重参数更加接近0 训练集误差有所增大 不过效果还好


# 简洁实现
def fit_and_plot_gluon(wd):
    # 网络定义和初始化
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))

    # 进对权重衰减 权重名字一般以weight结尾
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd', {'learning_rate':lr, 'wd':wd})
    # 不对偏差衰减 权重名字一般以bias结尾
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd', {'learning_rate':lr})

    train_l , test_l = [], []
    for _ in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            # 对两个Trainer实例分别调用step,分别得到更新权重和偏差
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_l.append(loss(net(train_features), train_labels).mean().asscalar())
        test_l.append(loss(net(test_features), test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs+1), train_l, 'epochs', 'loss',
                    range(1, num_epochs+1), test_l, ['train', 'test'], figsize=(15,5))

    print('L2 norm of w:', net[0].weight.data().norm().asscalar())


# fit_and_plot_gluon(wd=0)
# fit_and_plot_gluon(wd=3)

