import d2lzh as d2l
from mxnet import nd
import numpy as np
import math

#################################################  一维情况 #####################################################
# TODO y=x^2, gd下降
def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x                    # 梯度下降
        results.append(x)                   # 返回x的轨迹
    print('epoch 10, x:', x)
    return results
# Test
res = gd(0.2)


def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)                  # [-10,20]

    d2l.set_figsize(figsize=(15,5))
    d2l.plt.plot(f_line, [x**2 for x in f_line])    # y=x^2
    d2l.plt.plot(res, [x**2 for x in res], '-o')    # x的轨迹
    # 坐标轴
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')


show_trace(res)



################################################# 高维情况 #####################################################
# TODO y=x1^2+2*x2^2
def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0                                   #  s1, s2自变量状态,这里没用
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1,x2))
    print('epoch %d, (%f, %f)' % (i+1, x1, x2))
    return results

def show_trace2D(fx, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')             # (x1,x2)的轨迹
    x1, x2 = np.meshgrid(np.arange(-5.5,1.0, 0.1), np.arange(-3.0,1.0, 0.1))    # 坐标轴
    d2l.plt.contour(x1, x2, fx(x1,x2), colors='#1f77b4')            # 等高线
    # 坐标轴
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

eta = 0.05
def fx_2d(x1, x2):              # f(x)
    return x1**2 + 2*x2**2

def gd_2d(x1, x2, s1, s2):      # 梯度
    return(x1-eta*2*x1, x2-eta*4*x2, 0, 0)

d2l.plt.figure()
show_trace2D(fx_2d, train_2d(gd_2d)) # 画出轨迹图


################################################# 随机梯度下降 #####################################################
def gd_2d(x1, x2, s1, s2):      # 梯度
    return(x1 - eta *(2*x1 + np.random.normal(0.1)), 
            x2 - eta * (4*x2 + np.random.normal(0.1)), 0, 0)

# 随机梯度下降更加曲折
d2l.plt.figure()
show_trace2D(fx_2d, train_2d(gd_2d)) # 画出轨迹图


d2l.plt.show()

