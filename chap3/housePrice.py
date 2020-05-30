import d2lzh as d2l
from mxnet import autograd, init, nd, gluon
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd

# 读取数据
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
#print(train_data.shape); print(test_data.shape)    # (1460, 81) (1459, 80)
print(train_data.iloc[0:4, [0,1,2,3,-3,-2,-1]])

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features.shape)
#　找出取值是数的title
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 标准化数据　数据减去均值除标准差
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x-x.mean())/(x.std()))
# 标准化化后 所有的特征值变为0 直接用0替代缺损值(原始局显示为NA)
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 将离散值的特征转换成多个特征且只有一个取值为１
all_features = pd.get_dummies(all_features, dummy_na=True)      # dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
print('all_features.shape: ', all_features.shape)                                              # (2919, 331)

# 生成训练测试数据
n_train = train_data.shape[0]       # 0:行　１:列 这里1460
train_features = nd.array(all_features[:n_train].values)        # all_features[:n_train].values 把数据抽出来弄成NDArray
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))   # 售卖价格转成一维列向量

# 定义网络并初始化
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net

# 优化成对数均方根误差
loss = gloss.L2Loss()
def log_rmse(net, features, labels):
    # 将小于1的值改成1　取对数数值更加稳定
    clipped_preds = nd.clip(net(features), 1, float('inf'))    # 将数据限制在1-inf
    rmse = nd.sqrt(2*loss(clipped_preds.log(), labels.log()).mean())    # L2Loss有个1/2...
    return rmse.asscalar()                                      # 弄成标量

#　训练数据
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr, wd, batch_size):
    train_l, test_l = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate':lr, 'wd':wd})    # 使用adam优化算法

    for _ in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_l.append(log_rmse(net, train_features, train_labels))         # 一遍数据打包成一个结果
        if test_labels is not None:
            test_l.append(log_rmse(net, test_features, test_labels))
    return train_l, test_l


# K折交叉验证　将数据分成K份, 其中i是validate, 其余k-1份合在一起作为train
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k     # 整除k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size)   # idx是一个范围
        X_part, y_part = X[idx, :], y[idx]          # idx指定范围的数据拿出来
        if j == i:      # 找到牧宝块
            X_valid, y_valid = X_part, y_part
        elif X_train is None:       # 除了i,第一次执行回到这儿 后面全部链接到X_train, y_train
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)     # 0 将X_part放在X_train下面！再赋值给X_train
            y_train = nd.concat(y_train, y_part, dim=0)     # 1 将X_part放在X_train右面！这里没有使用！
    return X_train, y_train, X_valid, y_valid


# K折意味着跑ｋ次　test数据块轮流换
def k_fold(k, X_train, y_train, num_epochs, lr, wd, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_l, valid_l = train(net, *data, num_epochs, lr, wd, batch_size)
        train_l_sum += train_l[-1]
        valid_l_sum += valid_l[-1]
        # if i==0:
        #     d2l.semilogy(range(1,num_epochs+1), train_l, 'epochs', 'rmse',
        #                  range(1,num_epochs+1), valid_l, ['train_loss', 'valid_loss'], figsize=(15,5))
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_l[-1], valid_l[-1]))
    return train_l_sum/k, valid_l_sum/k


k, num_epochs, lr, wd, batch_size = 5, 300, 20, 0, 64
# k=5　意味着1460//5=292一块
train_l, test_l = k_fold(k, train_features, train_labels, num_epochs, lr, wd, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, test_l))


def train_and_pred(train_features, test_features, train_labels, test_labels, num_epochs, lr, wd, batch_size):
    net = get_net()
    train_l, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, wd, batch_size)
    # d2l.semilogy(range(1, num_epochs+1), train_l, 'epochs', 'rmes')
    print('train rmse %f' % train_l[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, wd, batch_size)










