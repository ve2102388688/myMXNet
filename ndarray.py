import mxnet
import numpy as np
from mxnet import autograd, nd
from time import time

print(mxnet.__version__);

x = nd.arange(12);
print(x);
print(x.shape)
print(x.size)

X = x.reshape((3, 4));      # reshape((3, -1))或reshape((-1, 4))
print(X)

y = nd.zeros((2, 3, 4));
print(y);

y = nd.ones((2, 3, 4));
print(y);

Y = nd.array([[2,1,4,3], [1,2,3,4], [4,3,2,1]]);
print(Y);

#Y = nd.random.normal(0,1, shape=(3, 4));
#print(Y);

print(X+Y);
print(X*Y);
print(X/Y);
print(Y.exp());
print(nd.dot(X, Y.T));
print(nd.concat(X, Y, dim=0), nd.concat(X, Y, dim=1));
print(X < Y);              # 如果X和Y在相同位置的条件判断为真(值相等),那么新的NDArray在相同位置的值为1;反之为0。
print(X.sum());
print(X.sum().asscalar())   # 换成标量

A = nd.arange(3).reshape((3, 1));
B = nd.arange(2).reshape((1, 2));
print(A+B)

print(X[1:3]);
X[1, 2] = 9;
print(X)

X[1:2, :] = 12;
print(X);

# 运算前后有不同的物理地址
before = id(Y);
Y = Y + X;
print(id(Y) == before);

Z = Y.zeros_like();
before = id(Z);
Z[:] = Y + X;
print(id(Z) == before);

nd.elemwise_add(X, Y, out=Z);       # 不会导致申请临时内存
print(id(Z) == before);

# numpy和NDarray转换
p = np.ones((2, 3));
print(p);
d = nd.array(p)
print(d);
print(d.asnumpy());


# 自动梯度求导
x1 = nd.arange(4).reshape((4, 1));
print(x1);
x1.attach_grad();               # 申请存放结果的内存，置0
with autograd.record():         # 记录问题描述
    y = 2*nd.dot(x1.T, x1);
y.backward();                   # 自动求导-4*x1
print(x1.grad)                  # 带入初始值x1
assert (x1.grad-4*x1).norm().asscalar() == 0

print(autograd.is_training());          # 预测模式
with autograd.record():
    print(autograd.is_training());      # 训练模式

def f(a):
    b = a*2;
    while b.norm().asscalar() < 1000 :
        b = b*2;
    if b.sum().asscalar() > 0:
        c = b;
    else:
        c = 100*b;
    return c;

a = nd.random.normal(0, 1, shape=(3, 4));
print(a);
a.attach_grad();
with autograd.record():
    c = f(a);
c.backward();
print(a.grad);
print(a.grad == c/a);


print(dir(nd.random));
help(nd.ones_like);

# 测试矢量计算
a = nd.ones(shape=1000)     # 行向量
b = nd.ones(shape=1000)     # 行向量

start = time();
c = nd.zeros(shape=1000)
for i in range(1000):
    c[i] = a[i] + b[i];
print(time()-start);


start = time();
d = a + b;
print(time()-start);




