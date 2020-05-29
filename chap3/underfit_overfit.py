import d2lzh as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss, data as gdata, nn
# import matplotlib
# print(matplotlib.matplotlib_fname())          # matplotlib字体配置文件路径

# 中问字体显示
d2l.plt.rcParams['font.sans-serif']=['SimHei']
d2l.plt.rcParams['axes.unicode_minus']=False

# y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + noise
# 假定测试样本和训练样本都是100
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = nd.random.normal(shape=(n_test+n_train, 1))                             # 前面100个训练 后面100测试
poly_features = nd.concat(features, nd.power(features,2), nd.power(features,3))    # [x x^2 x^3]
labels = nd.dot(poly_features, nd.transpose(nd.array(true_w))) + true_b            # y
labels += nd.random.normal(scale=0.1, shape=labels.shape)
print(features[:2], poly_features[:2], labels[:2])


# 做图函数 y轴使用对数坐标
def semilogy(x_vals, y_vals, x_label, y_label, title, x2_vals=None, y2_vals=None, legend=None, figsize=(15,5)):
    d2l.set_figsize(figsize)               # 显示大小
    d2l.plt.figure()                       # 开一个空图片
    d2l.plt.title(title)                   # 加上标题
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)       # 曲线1
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')  # 曲线2
        d2l.plt.legend(legend)                             # 曲线标注

# 拟合函数并画图
num_epochs, loss = 100, gloss.L2Loss()
def fit_and_plot(train_features, test_features, train_labels, test_labels, title=None):
    # 定义单层网络并初始化
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    # 训练集 学习器
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.01})

    train_l, test_l = [], []
    for _ in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_l.append(loss(net(train_features), train_labels).mean().asscalar())      # 每次100个loss求平均 进行120次
        # print(loss(net(train_features), train_labels))
        test_l.append(loss(net(test_features), test_labels).mean().asscalar())
    # print(len(train_l), train_l) print(len(test_l), test_l)
    print('final epoch: train_loss', train_l[-1], ', test loss', test_l[-1])             # 输出最后一次的train_loss test_loss
    semilogy(range(1, num_epochs+1), train_l, 'epochs', 'loss', title,
             range(1, num_epochs+1), test_l, ['train', 'test'])
    print('weight:', net[0].weight.data().asnumpy(), '\nbias:', net[0].bias.data().asnumpy())  # 输出网络参数



# 正常拟合
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], '正常拟合')

# 模型太简单了 欠拟合
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:], '欠拟合')

# 训练样本不足 过常拟合
fit_and_plot(poly_features[:2, :], poly_features[n_train:, :], labels[:2], labels[n_train:], '过拟合')

d2l.plt.show()


