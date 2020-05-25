from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random


# 生成数据集
num_inputs = 2;                                                         # 特征数
num_examples = 1000;                                                    # 样本数
true_w = [2, -3.4];
true_b = 4.2;
X = nd.random.normal(scale=1, shape=(num_examples, num_inputs));        # 1000*2
y = nd.dot(X, nd.transpose(nd.array(true_w))) + true_b;                 # y = X*W + b
y += nd.random.normal(0, 0.01, shape=y.shape);                          # y = X*W + b + noise
print(X[0], y[0]);


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg');
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display();
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize;

set_figsize(figsize=(10, 5));
plt.scatter(X[:, 1].asnumpy(), y.asnumpy(), 1);
# plt.show();                   # 这一句要有，不然不显示图片



def data_iter(batch_size, features, labels):
    num_examples = len(features);
    indices =list(range(num_examples));
    random.shuffle(indices);
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i+batch_size, num_examples)]);
        yield features.take(j), labels.take(j);         # take根据索引返回具体元素

batch_size = 10;
# for features,labels in data_iter(batch_size, X, y):
#     print(features, labels);
#     break;


# 定义paramenter
w = nd.random.normal(0, 0.01, shape=(num_inputs, 1));
b = nd.zeros(shape=(1,));
# 预先分配内存，不然后面backward异常
w.attach_grad();
b.attach_grad();

# 定义线性回归网络
def linearR(X, w, b):
    return nd.dot(X, w) + b;

# 定义平方损失函数
def square_loss(yHat, y):
    return (yHat - y.reshape(yHat.shape)) ** 2 / 2;

# 定义随机梯度下降
def GSD(parameters, lr, batch_size):
    for parameter in parameters:
        parameter[:] = parameter - lr*parameter.grad / batch_size;

# train samples
lr = 1.9;
num_epochs = 5;     # 跑多少次training data

for epoch in range(num_epochs):
    for features,labels in data_iter(batch_size, X, y):
        with autograd.record():
            loss = square_loss(linearR(features, w, b), labels);            # 用features,labels训练w,b
        loss.backward();                                                    # 求梯度
        GSD([w, b], lr, batch_size);
    train_loss  = square_loss(linearR(X, w, b), y);                         # 用模型w,b去验证误差
    print('epoch=%d, train_loss=%f' % (epoch+1, train_loss.mean().asnumpy()));

print('\nresult:');
print(true_w, w);                       # 真实w 训练的w
print(true_b, b);                       # 真实b 训练的b

